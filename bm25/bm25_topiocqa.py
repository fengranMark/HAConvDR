from re import T
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import argparse
import os
from utils import check_dir_exist_or_build
from os import path
from os.path import join as oj
import toml
import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval

def main():
    args = get_args()
    
    query_list = []
    qid_list = []
    with open(args.input_query_path, "r") as f:
        data = f.readlines()

    with open(args.input_query_path_2, "r") as f2:
        data_2 = f2.readlines()


    n = len(data)

    if args.use_PRF:
        with open(args.PRF_file, 'r') as f:
            PRF = f.readlines()
        assert(len(data) == len(PRF))

        if args.level == "token+turn":
            with open(args.PRF_file_2, 'r') as f:
                PRF_2 = f.readlines()
            assert(len(data) == len(PRF_2))
    for i in range(n):
        record = json.loads(data[i])
        if args.query_type == "raw":
            query = data[i]["query"]
        elif args.query_type == "rewrite":
            #query = data[i]['output']
            query = data[i]['rewrite'] #+ ' ' + data[i]['answer']
        elif args.query_type == "convq":
            query = ''
            if args.use_PRF:
                PRF[i] = json.loads(PRF[i])
                rel_label = PRF[i]["rel_label"]
                if len(rel_label) > 0:
                    history_query = data[i]["history_query"]

                    if args.level == "token":
                        token_set = [] # use for token-level expansion
                        for key in history_query:
                            sent = key.strip().split()
                            token_set.extend(sent)
                        for j in range(len(rel_label)):
                            if rel_label[j] == 1:
                                query += token_set[j] + ' '
                    elif args.level == "turn":
                        for j in range(len(rel_label)-1, -1, -1):
                            if rel_label[j] == 1:
                                query += data[i]["history_query"][j] + ' '
                    elif args.level == "token+turn":
                        PRF_2[i] = json.loads(PRF_2[i])
                        rel_label_2 = PRF_2[i]["rel_label"]
                        for j in range(len(rel_label)-1, -1, -1):
                            if rel_label[j] == 1:
                                query += history_query[j] + ' '

                        token_set = [] # use for token-level expansion
                        for key in history_query:
                            #if rel_label[j] == 0:
                            sent = key.strip().split()
                            token_set.extend(sent)
                        for j in range(len(rel_label_2)):
                            if rel_label_2[j] == 1:
                                query += token_set[j] + ' '
                    elif args.level == "token+turn":
                        PRF_2[i] = json.loads(PRF_2[i])
                        rel_label_2 = PRF_2[i]["rel_label"]

            else:
                for x in data[i]['history_query']:
                    query += x + ' '
            query = data[i]["query"] + query

        elif args.query_type == "convqa":
            query = ''
            if args.use_PRF:
                PRF[i] = json.loads(PRF[i])
                rel_label = PRF[i]["rel_label"]
                if len(rel_label) > 0:
                    for j in range(len(rel_label)-1, -1, -1):
                        if rel_label[j] == 1:
                            query += data[i]["history_query"][j] + ' '
                            query += data[i]["history_answer"][j] + ' '
            else:
                for j in range(len(data[i]['history_query'])):
                    query += data[i]['history_query'][j] + ' '
                    query += data[i]['history_answer'][j] + ' '
            query = data[i]["query"] + query

            query = query.strip().split()
            if len(query) > 512:
                query = query[-510:]
            query = ' '.join(query)

        elif args.query_type == "convqp":
            query = ""
            if args.use_PRF:
                PRF[i] = json.loads(PRF[i])
                rel_label = PRF[i]["rel_label"]
                if len(rel_label) > 0:
                    for j in range(len(rel_label)-1, -1, -1):
                        if rel_label[j] == 1:
                            query += record["history_query"][j] + ' '
                            query += json.loads(data[i - (len(rel_label) - j)])["pos_docs"][0] + ' '
                query += record["query"]
            else:
                for j in range(len(record['history_query'])):
                    query += record['history_query'][j] + ' '
                query = query + record["query"] + ' ' + record['last_response']
            query = query.strip().split()
            if len(query) > 512:
                query = query[-510:]
            query = ' '.join(query)

        elif args.query_type == "decode":
            query = record['oracle_utt_text']
            if args.eval_type == "answer":
                data_2[i] = json.loads(data_2[i])
                query = data_2[i]['answer_utt_text']
            elif args.eval_type == "oracle+answer":
                data_2[i] = json.loads(data_2[i])
                query = query + ' ' + data_2[i]['answer_utt_text']

        query_list.append(query)
        if "sample_id" in record:
            qid_list.append(record['sample_id'])
        else:
            qid_list.append(record['id'])
        
   
    # pyserini search
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k = args.top_k, threads = 20)

    total = 0
    with open(oj(args.output_dir_path, "bm25_train_oracle_res.trec"), "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                f.write("{} {} {} {} {} {} {}".format(qid,
                                                "Q0",
                                                item.docid[3:],
                                                i+1,
                                                -i - 1 + 200,
                                                item.score,
                                                "bm25"
                                                ))
                f.write('\n')
                total += 1
    print(total)


def print_res(run_file, qrel_file, rel_threshold):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}

    query_id = []

    
    for line in qrel_data:
        line = line.strip().split()
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
            query_id.append(query)
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    context_affect(query_id, mrr_list)

    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "NDCG@3": np.average(ndcg_3_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list), 
        }

    
    logger.info("---------------------Evaluation results:---------------------")    
    logger.info(res)
    return res

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        type = str,
                        required = True,
                        help = "Config file path.")

    args = parser.parse_args()
    config = toml.load(args.config)
    args = argparse.Namespace(**config)

    check_dir_exist_or_build([args.output_dir_path])
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    return args

if __name__ == '__main__':
    main()

# python bm25_topiocqa.py --config Config/bm25_topiocqa.toml
