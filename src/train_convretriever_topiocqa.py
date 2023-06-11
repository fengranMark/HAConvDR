import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import time
import copy
import pickle
import random
import numpy as np
import csv
import argparse
import toml
import os

from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from transformers import RobertaConfig, RobertaTokenizer
from models import ANCE
from utils import check_dir_exist_or_build, pstore, pload, set_seed, get_optimizer, print_res
from data import Retrieval_topiocqa, Retrieval_topiocqa_new


def save_model(args, model, tokenizer, step):
    if args.use_PRL:
        output_dir = oj(args.model_output_path, 'imp-bs{}-{}-goldPRL-{}preposhard-{}prepos-best-retriever-PRF'.format(str(args.per_gpu_train_batch_size), args.mode, str(args.is_prepos_neg), str(args.is_pseudo_prepos)))
    else:
        output_dir = oj(args.model_output_path, 'imp-bs{}-{}-noPRL-{}preposhard-{}prepos-best-retriever'.format(str(args.per_gpu_train_batch_size), args.mode, str(args.is_prepos_neg), str(args.is_pseudo_prepos)))
    #output_dir = oj(args.model_output_path, 'imp-bs{}-{}-goldPRL-MSE-best-retriever'.format(str(args.per_gpu_train_batch_size), args.mode, str(args.is_prepos_neg), str(args.is_pseudo_prepos)))
    #check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Step {}, Save checkpoint at {}".format(step, output_dir))

def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs):
    batch_size = len(query_embs)
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    if args.hard_neg_type != None:
        neg_scores = torch.sum(query_embs * neg_doc_embs, dim = 1).unsqueeze(1) # B * 1 hard negatives
        score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + 1)  in_batch negatives + 1 BM25 hard negative 
    else:
        score_mat = pos_scores
    label_mat = torch.arange(batch_size).to(args.device) # B
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)
    return loss

def cal_ranking_loss_prepos(query_embs, pos_doc_embs, bm25_neg_doc_embs, pseudo_prepos_embs=None, prepos_neg_doc_embs=None):
    batch_size = len(query_embs)
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    score_mat = pos_scores
    if args.is_pseudo_prepos and pseudo_prepos_embs != None:
        pseudo_pos_scores = query_embs.mm(pseudo_prepos_embs.T) # B * B
        score_mat = pos_scores + args.alpha * pseudo_pos_scores
    bm25_neg_scores = torch.sum(query_embs * bm25_neg_doc_embs, dim = 1).unsqueeze(1) # B * 1 bm25hard negatives
    score_mat = torch.cat([score_mat, bm25_neg_scores], dim = 1)    # B * (B + 1)  in_batch negatives + 1 BM25 hard negative 
    if args.is_prepos_neg and prepos_neg_doc_embs != None:
        prepos_neg_scores = torch.sum(query_embs * prepos_neg_doc_embs, dim = 1).unsqueeze(1) # B * 1 preposhard negatives
        score_mat = torch.cat([score_mat, prepos_neg_scores], dim = 1)    # B * (B + 2)  
    label_mat = torch.arange(batch_size).to(args.device) # B
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)
    return loss

def cal_kd_loss(query_embs, kd_embs):
    loss_func = nn.MSELoss()
    return loss_func(query_embs, kd_embs)

def train(args):
    # load the pretrained passage encoder model, but it will be frozen when training.
    # load conversational query encoder model
    config = RobertaConfig.from_pretrained(args.pretrained_encoder_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_encoder_path, do_lower_case=True)
    query_encoder = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(args.device)
    passage_encoder = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(args.device)

    if args.n_gpu > 1:
        query_encoder = torch.nn.DataParallel(query_encoder, device_ids = list(range(args.n_gpu)))

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    # data prepare
    train_dataset = Retrieval_topiocqa_new(args, tokenizer, args.train_file_path)
    train_loader = DataLoader(train_dataset, 
                                batch_size = args.batch_size, 
                                shuffle=True, 
                                collate_fn=train_dataset.get_collate_fn(args))

    logger.info("train samples num = {}".format(len(train_dataset)))
    
    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
    num_warmup_steps = args.num_warmup_portion * total_training_steps
    
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)

    global_step = 0

    # begin to train
    logger.info("Start training...")
    logger.info("Total training epochs = {}".format(args.num_train_epochs))
    logger.info("Total training steps = {}".format(total_training_steps))
    
    num_steps_per_epoch = total_training_steps // args.num_train_epochs
    logger.info("Num steps per epoch = {}".format(num_steps_per_epoch))

    epoch_iterator = trange(args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)

    best_loss = 1000
    total_loss = 0
    accumulated_loss = 0
    for epoch in epoch_iterator:
        query_encoder.train()
        passage_encoder.eval()
        for batch in tqdm(train_loader,  desc="Step", disable=args.disable_tqdm):
            query_encoder.zero_grad()
            bt_sample_ids = batch["bt_sample_ids"] # question id
            if args.mode == "rewrite":
                input_ids = batch["bt_rewrite"].to(args.device)
                input_masks = batch["bt_rewrite_mask"].to(args.device)
            elif args.mode == "raw":
                input_ids = batch["bt_raw_query"].to(args.device)
                input_masks = batch["bt_raw_query_mask"].to(args.device)
            elif args.mode == "convq":
                input_ids = batch["bt_conv_q"].to(args.device)
                input_masks = batch["bt_conv_q_mask"].to(args.device)
            elif args.mode == "convqa":
                input_ids = batch["bt_conv_qa"].to(args.device)
                input_masks = batch["bt_conv_qa_mask"].to(args.device)
            elif args.mode == "convqp":
                input_ids = batch["bt_conv_qp"].to(args.device)
                input_masks = batch["bt_conv_qp_mask"].to(args.device)
            else:
                raise ValueError("args.mode :{}, has not been implemented.".format(args.mode))

            bt_pos_docs = batch['bt_pos_docs'].to(args.device) # B * len one pos
            bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(args.device)
            bt_neg_docs = batch['bt_neg_docs'].to(args.device) # B * len batch size negs
            bt_neg_docs_mask = batch['bt_neg_docs_mask'].to(args.device)
            bt_pseudo_prepos_docs = batch['bt_pseudo_prepos_docs'].to(args.device)
            bt_pseudo_prepos_docs_mask = batch['bt_pseudo_prepos_docs_mask'].to(args.device)
            bt_prepos_neg_docs = batch['bt_prepos_neg_docs'].to(args.device)
            bt_prepos_neg_docs_mask = batch['bt_prepos_neg_docs_mask'].to(args.device)

            conv_query_embs = query_encoder(input_ids, input_masks)  # B * dim
            pseudo_prepos_embs, prepos_neg_docs_embs = None, None
            with torch.no_grad():
                # freeze passage encoder's parameters
                pos_doc_embs = passage_encoder(bt_pos_docs, bt_pos_docs_mask).detach()  # B * dim
                neg_doc_embs = passage_encoder(bt_neg_docs, bt_neg_docs_mask).detach()  # B * dim, hard negative
                if len(batch['bt_pseudo_prepos_docs'][0]) > 0:
                    #breakpoint()
                    # if bt_pseudo_prepos_docs.shape[1] != args.max_concat_length:
                    #     batch_size, ratio, seq_len = bt_pseudo_prepos_docs.shape       
                    #     bt_pseudo_prepos_docs = bt_pseudo_prepos_docs.view(batch_size * ratio, seq_len)        
                    #     bt_pseudo_prepos_docs_mask = bt_pseudo_prepos_docs_mask.view(batch_size * ratio, seq_len)             
                    #else:
                    pseudo_prepos_embs = passage_encoder(bt_pseudo_prepos_docs, bt_pseudo_prepos_docs_mask).detach()
                if len(batch['bt_prepos_neg_docs'][0]) > 0:
                    # if bt_prepos_neg_docs.shape[1] != args.max_concat_length:
                    #     batch_size, ratio, seq_len = bt_prepos_neg_docs.shape       
                    #     bt_prepos_neg_docs = bt_prepos_neg_docs.view(batch_size * ratio, seq_len)        
                    #     bt_prepos_neg_docs_mask = bt_prepos_neg_docs_mask.view(batch_size * ratio, seq_len)
                    #else:
                    #breakpoint()
                    prepos_neg_docs_embs = passage_encoder(bt_prepos_neg_docs, bt_prepos_neg_docs_mask).detach()

            #loss = cal_ranking_loss(conv_query_embs, pos_doc_embs, neg_doc_embs)
            loss = cal_ranking_loss_prepos(conv_query_embs, pos_doc_embs, neg_doc_embs, pseudo_prepos_embs, prepos_neg_docs_embs)
            #if args.use_mse and pseudo_prepos_embs != None:
            #    mse_loss = cal_kd_loss(pos_doc_embs, pseudo_prepos_embs)
            #    loss = args.beta * mse_loss + loss
            loss.backward()

            total_loss += loss.sum().item()   
            accumulated_loss += loss.sum().item() 
            global_step += 1
            
            if args.print_steps > 0 and global_step % args.print_steps == 0:
                logger.info("Epoch = {}, Global Step = {}, Step Loss = {}, Accumulated Loss = {}, Total Loss = {}".format(
                                epoch + 1,
                                global_step,
                                loss.item(),
                                accumulated_loss,
                                total_loss))

            if global_step % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                accumulated_loss = 0
                query_encoder.zero_grad()

            if best_loss > loss:
                save_model(args, query_encoder, tokenizer, global_step)
                best_loss = loss

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_encoder_path", type=str, default="checkpoints/ad-hoc-ance-msmarco")
    parser.add_argument("--train_file_path", type=str, default="datasets/topiocqa/train_with_info_new.json")
    parser.add_argument("--collection", type=str, default="datasets/topiocqa/full_wiki_segments.tsv")
    parser.add_argument('--model_output_path', type=str, default="../output/topiocqa/model")    
    
    parser.add_argument("--num_train_epochs", type=int, default=5, help="num_train_epochs")
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=256, help="Max doc length, consistent with \"Dialog inpainter\".")
    parser.add_argument("--max_response_length", type=int, default=32)
    parser.add_argument("--max_concat_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--disable_tqdm", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="convqp")
    parser.add_argument("--use_PRL", type=bool, default=True)
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--is_prepos_neg", type=bool, default=True)
    parser.add_argument("--is_pseudo_prepos", type=bool, default=False)
    parser.add_argument("--is_PRF", type=bool, default=True)
    parser.add_argument("--per_gpu_train_batch_size", type=int,  default=8)
    parser.add_argument("--alpha", type=float,  default=1)
    #parser.add_argument("--beta", type=float,  default=0.5)

    parser.add_argument("--print_steps", type=float, default=64)
    parser.add_argument("--accumulation_steps", type=float, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--num_warmup_portion", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    args = parser.parse_args()

    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    return args

if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    train(args)
