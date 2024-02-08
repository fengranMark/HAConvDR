from json.tool import main
import json
from tqdm import tqdm, trange
import csv
import random

def gen_topiocqa_qrel(raw_dev_file_path, output_qrel_file_path):
    '''
    raw_dev_file_path = "gold_dev.json"
    output_qrel_file_path = "topiocqa_qrel.trec"
    '''
    with open(raw_dev_file_path, "r") as f:
        data = json.load(f)
    
    with open(output_qrel_file_path, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("TopiOCQA-Dev", line["conv_id"], line["turn_id"])
            for pos in line["positive_ctxs"]:
                #pid = int(pos["passage_id"]) - 1
                pid = int(pos["passage_id"])
                f.write("{} {} {} {}".format(sample_id, 0, pid, 1))
                f.write('\n')


def gen_train_test_files(raw_train_file_path, raw_dev_file_path, output_train_file_path, ouput_test_file_path, collection_file_path):
    '''
    raw_train_file_path = "gold_train.json"
    raw_dev_file_path = "gold_dev.json"
    output_train_file_path = "train.json"
    ouput_test_file_path = "test.json"
    collection_file_path = "full_wiki_segments.tsv"
    '''
    qid2passage = {}
    with open(collection_file_path, 'r') as fin:
        reader = csv.reader(fin, delimiter="\t")
        for i, row in enumerate(tqdm(reader)):
            if row[0] == "id": # ['id', 'text', 'title'] id begin from 1
                continue
            idx, text, title = int(row[0]), row[1], ' '.join(row[2].split(' [SEP] '))
            qid2passage[idx] = " ".join([title, text])

    with open(raw_train_file_path, "r") as f:
        data = json.load(f)
    
    last_conv_id = -1
    last_response = ""
    context_queries_and_answers = []
    context_pos_docs_pids = set()
    random_pid = list(range(25700592))

    with open(output_train_file_path, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("TopiOCQA-Train", line["conv_id"], line["turn_id"])
            query = line["question"]
            answers = line["answers"]
            if len(answers) == 0:
                answer = "UNANSWERABLE"
            else:
                answer = answers[0]

            positive_ctxs = line["positive_ctxs"]
            pos_docs = []
            pos_docs_pids = []
            for pos in positive_ctxs:
                passage = pos["title"].rstrip().replace(' [SEP] ', ' ') + ' ' + pos["text"].rstrip()
                pos_docs.append(passage)
                pos_docs_pids.append(int(pos["passage_id"]))            
            # hard_negative_ctxs = line["hard_negative_ctxs"]
            # negative_ctxs = line["negative_ctxs"]

            record = {}
            record["sample_id"] = sample_id
            record["cur_utt_text"] = query
            if int(line["conv_id"]) != last_conv_id:
                context_queries_and_answers = []
                context_pos_docs_pids = set()
                last_response = ""
            #record["ctx_utts_text"] = context_queries_and_answers
            record["last_response"] = last_response
            record["pos_docs"] = pos_docs
            record["pos_docs_pids"] = pos_docs_pids

            prepos_neg_docs_pids = list(context_pos_docs_pids - set(pos_docs_pids))
            neg_docs = []
            neg_docs_pids = []
            if len(prepos_neg_docs_pids):
                neg_docs_pids.append(random.choice(prepos_neg_docs_pids))
            else:
                neg_docs_pids.append(random.choice(random_pid))
            neg_docs.append(qid2passage[neg_docs_pids[0]])

            record["neg_docs"] = neg_docs
            record["neg_docs_pids"] = neg_docs_pids
            record["prepos_neg_docs_pids"] = prepos_neg_docs_pids
            f.write(json.dumps(record))
            f.write('\n')

            last_response = positive_ctxs[0]["title"].rstrip().replace(' [SEP] ', ' ') + ' ' + positive_ctxs[0]["text"].rstrip()
            context_pos_docs_pids |= set(pos_docs_pids)
            #context_queries_and_answers.append(query)
            #context_queries_and_answers.append(answer)
            last_conv_id = int(line["conv_id"])


    with open(raw_dev_file_path, "r") as f:
        data = json.load(f)
    
    last_conv_id = -1
    last_response = ""
    context_queries_and_answers = []
    with open(ouput_test_file_path, "w") as f:
        for line in tqdm(data):
            sample_id = "{}_{}_{}".format("TopiOCQA-Dev", line["conv_id"], line["turn_id"])
            query = line["question"]
            answers = line["answers"]
            if len(answers) == 0:
                answer = "UNANSWERABLE"
            else:
                answer = answers[0]

            positive_ctxs = line["positive_ctxs"]
            pos_docs = []
            pos_docs_pids = []
            for pos in positive_ctxs:
                passage = pos["title"].rstrip().replace(' [SEP] ', ' ') + ' ' + pos["text"].rstrip()
                pos_docs.append(passage)
                pos_docs_pids.append(int(pos["passage_id"]))
            # hard_negative_ctxs = line["hard_negative_ctxs"]
            # negative_ctxs = line["negative_ctxs"]

            record = {}
            record["sample_id"] = sample_id
            record["cur_utt_text"] = query
            if int(line["conv_id"]) != last_conv_id:
                context_queries_and_answers = []
                context_pos_docs_pids = set()
            #record["ctx_utts_text"] = context_queries_and_answers
            record["last_response"] = last_response
            record["pos_docs"] = pos_docs
            record["pos_docs_pids"] = pos_docs_pids

            prepos_neg_docs_pids = list(context_pos_docs_pids - set(pos_docs_pids))
            neg_docs = []
            neg_docs_pids = []
            if len(prepos_neg_docs_pids):
                neg_docs_pids.append(random.choice(prepos_neg_docs_pids))
            else:
                neg_docs_pids.append(random.choice(random_pid))
            neg_docs.append(qid2passage[neg_docs_pids[0]])

            record["neg_docs"] = neg_docs
            record["neg_docs_pids"] = neg_docs_pids
            record["prepos_neg_docs_pids"] = prepos_neg_docs_pids
            f.write(json.dumps(record))
            f.write('\n')

            last_response = positive_ctxs[0]["title"].rstrip().replace(' [SEP] ', ' ') + ' ' + positive_ctxs[0]["text"].rstrip()
            context_pos_docs_pids |= set(pos_docs_pids)
            #context_queries_and_answers.append(query)
            #context_queries_and_answers.append(answer)
            last_conv_id = int(line["conv_id"])

def merge_rel_label_info(rel_file, orig_file, new_file):
    # rel_file: train/dev_rel_label_rawq.json
    # orig_file: train/test.json
    # new_file: train/test_with_gold_rel.json
    with open(rel_file, "r") as f:
        rel_labels = f.readlines()

    with open(orig_file, 'r') as f, open(new_file, 'w') as g:
        lines = f.readlines()
        for i in range(len(lines)):
            line_dict = json.loads(lines[i])
            sample_id = line_dict['sample_id']
            if sample_id.split('-')[-1] != '1':
                assert sample_id == json.loads(rel_labels[i])['id']
                rel_label = json.loads(rel_labels[i])['rel_label']
                line_dict['rel_label'] = rel_label
            else:
                line_dict['rel_label'] = []
            json.dump(line_dict, g)
            g.write('\n')

def merge_bm25_neg_info(bm25_run_file, orig_file, new_file):
    qid2bm25_pid = {}
    with open(bm25_run_file, 'r') as f:
        data = f.readlines()

    for line in data:
        line = line.strip().split()
        qid, pid = line[0], int(line[2])
        if qid not in qid2bm25_pid:
            qid2bm25_pid[qid] = [pid]
        else:
            qid2bm25_pid[qid].append(pid)

    with open(orig_file, 'r') as f:
        ori_data = f.readlines()

    with open(new_file, 'w') as g:
        for line in ori_data:
            record = json.loads(line)
            qid = record["sample_id"]
            pos_docs_pids = record["pos_docs_pids"]
            bm25_hard_neg_docs_pids = []
            for pid in qid2bm25_pid[qid]:
                if pid not in pos_docs_pids:
                    bm25_hard_neg_docs_pids.append(pid)
            record["bm25_hard_neg_docs_pids"] = bm25_hard_neg_docs_pids
            g.write(json.dumps(record))
            g.write('\n')
        
            
def extract_doc_content_of_bm25_hard_negs_for_train_file(collection_path, 
                                                         train_inputfile, 
                                                         train_outputfile_with_doc,
                                                         neg_ratio=2):
    '''
    - collection_path = "collection.tsv"
    - train_inputfile = "train.json"
    - train_outputfile_with_doc = "train_with_neg.json"       
    '''
    with open(train_inputfile, 'r') as f:
        data = f.readlines()
    
    pid2passage = {}
    with open(collection_file_path, 'r') as fin:
        reader = csv.reader(fin, delimiter="\t")
        for i, row in enumerate(tqdm(reader)):
            if row[0] == "id": # ['id', 'text', 'title'] id begin from 1
                continue
            idx, text, title = int(row[0]), row[1], ' '.join(row[2].split(' [SEP] '))
            pid2passage[idx] = " ".join([title, text])
    
    # Merge doc content to the train file
    with open(train_outputfile_with_doc, 'w') as fw:
        for line in data:
            line = json.loads(line)
            pos_docs_pids = line["pos_docs_pids"]
            neg_docs_text = []
            for pid in line["bm25_hard_neg_docs_pids"]: #[:neg_ratio]:
                if pid in pid2passage and pid not in pos_docs_pids:
                    neg_docs_text.append(pid2passage[pid])
            
            line["bm25_hard_neg_docs"] = neg_docs_text
            
            fw.write(json.dumps(line))
            fw.write('\n')

         

def modify_pos_docs(conv_sample, pos_docs_text):
    '''
    Modify the pos doc content based on the current conversational sample 
    to avoid simply string-match, enhance model generalization ability
    '''
    return pos_docs_text

def modify_neg_docs(conv_sample, neg_docs_text):
    '''
    Modify the neg doc content based on the current conversational sample 
    to avoid simply string-match, enhance model generalization ability
    '''
    return neg_docs_text

def reformulate_dataset_info(input_file, output_file, add_file):
    with open(input_file, 'r') as f, open(output_file, 'w') as g, open(add_file, 'r') as add:
        data = f.readlines()
        add_data = add.readlines()

        for i in range(len(data)):
            record = json.loads(data[i])
            
            qid = record["sample_id"]
            cur_utt_text = record["cur_utt_text"]
            last_response = record["last_response"]
            pos_docs = record["pos_docs"]
            pos_docs_pids = record["pos_docs_pids"]
            prepos_docs_pids = record["prepos_neg_docs_pids"]
            rel_label = record["rel_label"]
            bm25_hard_neg_docs_pids = record["bm25_hard_neg_docs_pids"]
            bm25_hard_neg_docs = record["bm25_hard_neg_docs"]

            pseudo_prepos_docs_pids = []
            pseudo_prepos_docs = []
            prepos_neg_docs_pids = []
            prepos_neg_docs = []

            for idx, label in enumerate(rel_label):
                prepos_doc = json.loads(data[i - idx])["pos_docs"]
                prepos_doc_pid = json.loads(data[i - idx])["pos_docs_pids"]
                if label == 1:
                    pseudo_prepos_docs.extend(prepos_doc)
                    pseudo_prepos_docs_pids.extend(prepos_doc_pid)
                else:
                    prepos_neg_docs.extend(prepos_doc)
                    prepos_neg_docs_pids.extend(prepos_doc_pid)

            g.write(json.dumps({
                    "sample_id": qid,
                    "cur_utt_text": cur_utt_text,
                    "last_response": last_response,
                    "pos_docs": pos_docs,
                    "pos_docs_pids": pos_docs_pids,
                    "bm25_hard_neg_docs": bm25_hard_neg_docs,
                    "bm25_hard_neg_docs_pids": bm25_hard_neg_docs_pids,
                    "pseudo_prepos_docs": pseudo_prepos_docs,
                    "pseudo_prepos_docs_pids": pseudo_prepos_docs_pids,
                    "prepos_neg_docs": prepos_neg_docs,
                    "prepos_neg_docs_pids": prepos_neg_docs_pids,
                    "rel_label": rel_label,
                    }) + "\n")
        print("finish")

def select_pseudo_relevant_feedback_passage(bm25_trec_file, ance_trec_file, neg_ratio=3):
    with open(bm25_trec_file, 'r') as f, open(ance_trec_file) as g:
        bm25_data, ance_data = f.readlines(), g.readlines()

    qid2selected_pos, qid2selected_neg = {}, {}
    bm25_pid_list, ance_pid_list, cooc_pid = [], [], {} # pid: bm25_rank + ance_rank
    print(len(bm25_data), len(ance_data))
    for idx in trange(len(bm25_data)):
        bm25_line = bm25_data[idx].strip().split()
        ance_line = ance_data[idx].strip().split()
        bm25_qid, bm25_pid, bm25_rank = bm25_line[0], int(bm25_line[2]), int(bm25_line[3])
        ance_qid, ance_pid, ance_rank = ance_line[0], int(ance_line[2]), int(ance_line[3])
        assert bm25_qid == ance_qid
        assert bm25_rank == ance_rank
        qid = bm25_qid
        bm25_pid_list.append(bm25_pid)
        ance_pid_list.append(ance_pid)
        if ance_rank == 100:
            qid2selected_neg[qid], qid2selected_pos[qid] = [], []
            # for pos selection
            if len(set(bm25_pid_list) & set(ance_pid_list)) == 0:
                for i in range(neg_ratio):
                    qid2selected_pos[qid].append(ance_pid_list[i])
            for i in range(10):
                # for neg selection
                if bm25_pid_list[i] not in ance_pid_list and bm25_pid_list[i] not in qid2selected_pos[qid]:
                    qid2selected_neg[qid].append(bm25_pid_list[i])
                if ance_pid_list[i] not in bm25_pid_list and ance_pid_list[i] not in qid2selected_pos[qid]:
                    qid2selected_neg[qid].append(ance_pid_list[i])
            if len(qid2selected_neg[qid]) > neg_ratio:
                qid2selected_neg[qid] = qid2selected_neg[qid][:neg_ratio]
            # one rank higher and another rank lower for neg, both higher for pos                   
            for i in range(100):
                if bm25_pid_list[i] in ance_pid_list:
                    if bm25_pid_list[i] in cooc_pid:
                        cooc_pid[bm25_pid_list[i]] = min(i + ance_pid_list.index(bm25_pid_list[i]), cooc_pid[bm25_pid_list[i]])
                    else:
                        cooc_pid[bm25_pid_list[i]] = i + ance_pid_list.index(bm25_pid_list[i])
                if ance_pid_list[i] in bm25_pid_list:
                    if ance_pid_list[i] in cooc_pid:
                        cooc_pid[ance_pid_list[i]] = min(i + bm25_pid_list.index(ance_pid_list[i]), cooc_pid[ance_pid_list[i]])
                    else:
                        cooc_pid[ance_pid_list[i]] = i + bm25_pid_list.index(ance_pid_list[i])
            cooc_pid = sorted(cooc_pid.items(), key=lambda x: x[1], reverse=False)
            # for pos selection
            if len(qid2selected_pos[qid]) < neg_ratio:
                for i in range(len(cooc_pid)):
                    pid = cooc_pid[i][0]
                    qid2selected_pos[qid].append(pid)
                    if len(qid2selected_pos[qid]) == neg_ratio:
                        break 
            # for additional neg
            if len(qid2selected_neg[qid]) < neg_ratio:
                for i in range(len(cooc_pid) - 1, -1, -1):
                    pid = cooc_pid[i][0]
                    if pid not in qid2selected_pos[qid]:
                        qid2selected_neg[qid].append(pid)
                    if len(qid2selected_neg[qid]) == neg_ratio:
                        break    
            bm25_pid_list, ance_pid_list, cooc_pid = [], [], {}
    return qid2selected_pos, qid2selected_neg
        
def merge_pseudo_relevant_feedback(query_file, ance_trec_file, bm25_trec_file, collection_file, output_file):
    with open(ance_trec_file, 'r') as f:
        trec_data = f.readlines()
    qid2PRL_pos = {}
    for line in trec_data:
        line = line.strip().split()
        qid, pid, rank = line[0], int(line[2]), int(line[3])
        if rank > 3:
            continue
        if qid not in qid2PRL_pos:
            qid2PRL_pos[qid] = []
        qid2PRL_pos[qid].append(pid)    
    
    pid2passage = {}
    with open(collection_file_path, 'r') as fin:
        reader = csv.reader(fin, delimiter="\t")
        for i, row in enumerate(tqdm(reader)):
            if row[0] == "id": # ['id', 'text', 'title'] id begin from 1
                continue
            idx, text, title = int(row[0]), row[1], ' '.join(row[2].split(' [SEP] '))
            pid2passage[idx] = " ".join([title, text])
    
    qid2selected_pos, qid2selected_neg = select_pseudo_relevant_feedback_passage(bm25_trec_file, ance_trec_file)
    with open(query_file, 'r') as f, open(output_file, 'w') as g:
        query_data = f.readlines()
        for line in query_data:
            record = json.loads(line)
            qid = record['sample_id']
            # PRF pos
            PRF_pos_docs, PRF_pos_docs_pids = [], []
            for pid in qid2PRL_pos[qid]:
                PRF_pos_docs_pids.append(pid)
                PRF_pos_docs.append(pid2passage[pid])
            record["PRF_pos_docs"] = PRF_pos_docs
            record["PRF_pos_docs_pids"] = PRF_pos_docs_pids
            # selected pos and neg for PRF
            selected_PRF_pos_docs, selected_PRF_pos_docs_pids, selected_PRF_neg_docs, selected_PRF_neg_docs_pids = [], [], [], []
            for pid in qid2selected_pos[qid]:
                selected_PRF_pos_docs.append(pid2passage[pid])
                selected_PRF_pos_docs_pids.append(pid)
            for pid in qid2selected_neg[qid]:
                selected_PRF_neg_docs.append(pid2passage[pid])
                selected_PRF_neg_docs_pids.append(pid)
            record["selected_PRF_pos_docs"], record["selected_PRF_pos_docs_pids"] = selected_PRF_pos_docs, selected_PRF_pos_docs_pids
            record["selected_PRF_neg_docs"], record["selected_PRF_neg_docs_pids"] = selected_PRF_neg_docs, selected_PRF_neg_docs_pids
            g.write(json.dumps(record) + '\n')
            
def split_pos_passage_text(input_file, output_file):
    with open(input_file, 'r') as f:
        data = f.readlines()

    pass


if __name__ == "__main__":
    
    raw_train_file_path = "./gold_train.json" # data.gold_passages_info.all_history.train
    raw_dev_file_path = "./gold_dev.json" # data.gold_passages_info.all_history.dev
    output_train_file_path = "./train.json"
    output_test_file_path = "./test.json"
    collection_file_path = "./datasets/topiocqa/full_wiki_segments.tsv"
    gen_train_test_files(raw_train_file_path, raw_dev_file_path, output_train_file_path, output_test_file_path, collection_file_path)

    raw_dev_file_path = "./gold_dev.json"
    output_qrel_file_path = "./topiocqa_qrel.trec"
    gen_topiocqa_qrel(raw_dev_file_path, output_qrel_file_path)

    train_rel_file = "./output/topiocqa/dense_rel/train_rel_label.json"
    test_rel_file = "./output/topiocqa/dense_rel/dev_rel_label.json"
    train_new_file = "./train_with_gold_rel_p.json"
    test_new_file = "./test_with_gold_rel_p.json"
    merge_rel_label_info(train_rel_file, output_train_file_path, train_new_file)
    merge_rel_label_info(test_rel_file, output_test_file_path, test_new_file)

    bm25_run_file = "output/topiocqa/bm25/bm25_train_for_hardneg_res.trec"
    train_inputfile = "train_with_gold_rel_p.json"
    train_outputfile_with_doc = "train_with_gold_rel_p_neg.json"
    merge_bm25_neg_info(bm25_run_file, train_inputfile, train_outputfile_with_doc)
    extract_doc_content_of_bm25_hard_negs_for_train_file(collection_file_path, train_outputfile_with_doc, train_outputfile_with_doc)

    input_file = "train_with_gold_rel_p_neg.json"
    output_file = "train_with_info.json"
    add_file = "../../output/ANCE_goldPRL_train_convqp_PRF.trec"
    reformulate_dataset_info(input_file, output_file, add_file)
