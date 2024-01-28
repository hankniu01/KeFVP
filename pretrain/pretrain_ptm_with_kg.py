import os
from re import A
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

from time import strftime, localtime
import logging
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from sklearn.model_selection import train_test_split


from pretrain_models import PretrainPTMWithKG
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import pickle
import random, math
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import random, tqdm, sys, math, gzip
from data_utils_pretrain_with_kg import load_examples, TRAIN_SET, TEST_SET, DEV_SET, generate_dataset, set_seed
from transformers import AutoTokenizer, AutoModel, AdamW


class Dataset(data.Dataset):
    def __init__(self, texts, labels, labels_b):
        'Initialization'
        self.labels = labels
        self.text = texts
        self.labels_b = labels_b
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if torch.is_tensor(index):
            index = index.tolist()

        # Load data and get label
        X = self.text[index,:,:]
        y = self.labels[index]
        y_b = self.labels_b[index]
        return X, y, y_b


def get_bert_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(
    #     optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    return optimizer

def bert_freeze(args, bert):
    # if self.opt.bert_unfreeze != []:   # if bert_unfreeze == [], all bert params are not frozen
    for name, param in bert.named_parameters():
        param.requires_grad = False
        for pre_name in args.bert_unfreeze:
            if pre_name in name:
                param.requires_grad = True
                break
 
def go(args):
    """
    Creates and trains a basic transformer for the volatility regression task.
    """
    LOG2E = math.log2(math.e)
    NUM_CLS = 1
    
    set_seed(args.seed)
    
    args.tokenizer = AutoTokenizer.from_pretrained(args.ptm)
    
    args.logger.info(" Loading Data ...")
    
    train_data = load_examples(args, args.task_name, args.data_dir, TRAIN_SET, num_examples=args.train_examples)
    
    train_dataset = generate_dataset(args, train_data, labelled=True)
    train_sampler = data.RandomSampler(train_dataset)
    train_dataloader = data.DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    
    args.logger.info(" Finish Loading Data... ")

    # create the model
    model = PretrainPTMWithKG(args)
    model.ptm.resize_token_embeddings(len(args.tokenizer))
    if args.bert_unfreeze != []:   # if bert_unfreeze == [], all bert params are not frozen
        bert_freeze(args, model.ptm)
    
    if args.gpu:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.cuda()
        
    # opt = torch.optim.Adam(lr=args.lr, params=model.parameters(), weight_decay=args.weight_decay_for_no_bert)
    # training loop
    args.logger.info('pretrain mlm task ... ')
    pretrain(args, model, train_dataloader, task='mlm')


def pretrain(args, model, train_dataloader, task):
    
    opt = get_bert_optimizer(args, model)
    seen = 0
    best_loss = 100.0
    evaluation= {'Epoch': [],'Train MSE': [], 'Dev MSE':[], 'Dev MSE AUXILIARY':[], 'Test MSE':[], 'Test MSE AUXILIARY':[], 'Dev F1': [], \
                        'Dev ACC': [], 'Dev MATT': [], 'Test F1': [], 'Test ACC': [], 'Test MATT': []}
    for e in tqdm.tqdm(range(args.num_epochs)):
        train_loss_tol = 0.0
        print('\n epoch ',e)
        model.train()

        for i, batch in tqdm.tqdm(enumerate(train_dataloader)):
            # learning rate warmup
            # - we linearly increase the learning rate from 10e-10 to arg.lr over the first
            #   few thousand batches
            if args.lr_warmup > 0 and seen < args.lr_warmup:
                lr = max((args.lr / args.lr_warmup) * seen, 1e-10)
                opt.lr = lr

            opt.zero_grad()
            if args.gpu:
                batch = {k: t.cuda() for k, t in batch.items()}
            mlm_input_ids = batch['mlm_input_ids'].squeeze(1)
            is_ratio_input_ids = batch['sent_is_ratio'].squeeze(1)
            head_ids, rel_ids, tail_ids = batch['head_nd_input_ids'], batch['rel_nd_input_ids'], batch['tail_nd_input_ids']
            positon = batch['labels_is_ratio']
            posi = batch['position']

            labels = batch['labels_mlm']
            pred, transE_loss = model(mlm_input_ids, is_ratio_input_ids, head_ids, rel_ids, tail_ids, positon, posi, task)
            
            lb_bidx, lb_pidx = (labels != 0).nonzero(as_tuple=True)
            re_labels = labels[lb_bidx, lb_pidx]
            loss = nn.CrossEntropyLoss()(pred, re_labels) + transE_loss.mean(0) * 0.5
            
            train_loss_tol += loss
            
            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if args.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            
            opt.step()
            
            seen += mlm_input_ids.shape[0]
            args.logger.info('Epoch: {}, Train loss: {}'.format(e, train_loss_tol/(i+1)))
         
        train_loss_tol = train_loss_tol/(i+1)
        args.logger.info('Epoch: {}, Train loss: {}'.format(e, train_loss_tol))
        evaluation['Train MSE'].append(train_loss_tol.item())
        if train_loss_tol < best_loss:
            args.logger.info(['save model ... '])
            torch.save(model.state_dict(), '/your/project/path/pretrain_models/bert_base_uncased/pretrain_ptm.pkl')
            if torch.cuda.device_count() > 1:
                model.module.ptm.save_pretrained('/your/project/path/pretrain_models/bert_base_uncased')
            else:
                model.ptm.save_pretrained('/your/project/path/pretrain_models/bert_base_uncased')



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Argument for earning calls volatility prediction')
    parser.add_argument('--num_epochs', default=60, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--weight_decay_for_no_bert", default=0.001, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('--max_pool', default=False)   
    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--embedding_size', default=768, type=int)
    parser.add_argument('--gpu', default=True, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--lr_warmup', default=1000, type=int)
    parser.add_argument('--gradient_clipping', default=1.0, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--vocab_size', default=50000, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--pred_save_dir', default='text_dir', type=str)
    
    parser.add_argument('--saved_feature_dir', default='/your/project/path/save_features', type=str)
    parser.add_argument('--source', default='price', type=str)  # text&audio&price
    parser.add_argument('--ptm', default='bert-base-uncased', type=str)
    parser.add_argument('--ratio_max_length', default=200, type=int)   #213   250   
    parser.add_argument('--entity_max_length', default=50, type=int)   #213   250
    parser.add_argument('--run_mode', default='reg', type=str)  #cls
    parser.add_argument('--pid', default='0', type=str)
    parser.add_argument('--tf_type', default='kuma', type=str)
    parser.add_argument('--task_name', default='EarningCall', type=str)
    parser.add_argument("--data_dir", default='your/project/path/kept_dataset', type=str,     #data/k-shot/SST-2/16-13    data/original/SST-2
                    help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--train_examples", default=-1, type=int,
                    help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--eval_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--dev_examples", default=-1, type=int,
                        help="The total number of dev examples to use, where -1 equals all examples.")
    
    parser.add_argument('--feature_dir', default='/your/project/path/save_features')
    parser.add_argument('--duration', default=3, type=int, choices=[3, 7, 15, 30])
    parser.add_argument('--max_length', default=520, type=int)
    parser.add_argument('--max_num_sent', default=50, type=int)   # 256
    parser.add_argument("--batch_size", default=16, type=int,  help="Batch size")
    parser.add_argument('--price_lag', default=30, type=int)
    
    parser.add_argument('--dependent_z', default=False)
    parser.add_argument('--kuma_dropout', default=0.5, type=float)
    parser.add_argument('--kuma_hidden_size', default=128, type=int)
    
    parser.add_argument('--audio_embed_size', default=200, type=int)
    parser.add_argument('--audio_hd', default=8, type=int)
    parser.add_argument('--num_audio_layers', default=2, type=int)
    parser.add_argument('--audio_embed', default=26, type=int)
    
    parser.add_argument('--corss_hidden', default=100, type=int)
    parser.add_argument('--cross_head', default=8, type=int)
    
    parser.add_argument('--bert_unfreeze', type=list, default=['layer.10', 'layer.11'])
    
    
    args = parser.parse_args()
    
    logger = logging.getLogger('cli')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    log_file = '{}-{}.log'.format(args.task_name, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('/your/project/path/log', log_file)))
    args.logger = logger
    
    args.logger.info(args)
    
    go(args)
