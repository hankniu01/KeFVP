import os
from tqdm import tqdm
import argparse
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import json
from torch.utils.data import DataLoader


def generate_embedding(args, tokenizer, model):
    
    data_list = os.listdir(args.data_path)
    output_dct = {}
    all_sent_num = []
    for data_dir in tqdm(data_list):
        output_dct[data_dir] = {}
        # text_path = args.data_path + data_dir + '/text.txt'
        text_path = args.data_path + data_dir + '/Text.txt'   # For ec
        text_file = open(text_path)
        all_sent_for_one_text = []
        for line in text_file.readlines():
            line = line.strip()
            all_sent_for_one_text += [line]
        all_sent_num += [len(all_sent_for_one_text)]
        try:
            input = tokenizer(all_sent_for_one_text, padding=True, return_tensors='pt', truncation=True)
        except:
            print('wrong tokenize, sent_list_from: {}'.format(data_dir))
            continue
        with torch.no_grad():
            model_out = model(input['input_ids'].cuda(), input['attention_mask'].cuda())
            # model_out = model(input['input_ids'].cuda(), input['token_type_ids'].cuda(), input['attention_mask'].cuda())
        
        # last_hidden_out = model_out['last_hidden_state'].mean(1)
        # second_last_hidden_out = model_out['hidden_states'][-2].mean(1)
        out_for_on_text = model_out['pooler_output']
        num_sent, ptm_size = out_for_on_text.shape
        # for key, emb in {'last_hidden_out': last_hidden_out, 'pooler_output': out_for_on_text}.items():
        for key, emb in {'pooler_output': out_for_on_text}.items():
            if args.max_sent > num_sent:
                emb = torch.cat([emb, torch.zeros(args.max_sent - num_sent, ptm_size).cuda()], dim=0)
            else:
                emb = emb[:args.max_sent, :]
            output_dct[data_dir][key] = emb.cpu().detach().numpy()
    
    pickle.dump(output_dct, open(args.save_path, 'wb'))


def generate_embedding_large(args, tokenizer, model):
    
    data_list = os.listdir(args.data_path)
    output_dct = {}
    for data_dir in tqdm(data_list):
        output_dct[data_dir] = {}
        text_path = args.data_path + data_dir + '/Text.txt'
        text_file = open(text_path)
        all_sent_for_one_text = []
        last_hidden_out, out_for_on_text = [], []
        for line in text_file.readlines():
            line = line.strip()
            all_sent_for_one_text += [line]
        
        all_sent_for_one_text = DataLoader(all_sent_for_one_text, batch_size=32)
        for item in all_sent_for_one_text:
            try:
                input = tokenizer(item, padding=True, return_tensors='pt')
            except:
                print('wrong tokenize, sent_list_from: {}'.format(data_dir))
                continue
            with torch.no_grad():
                model_out = model(input['input_ids'].cuda(), input['token_type_ids'].cuda(), input['attention_mask'].cuda())
        
            last_hidden_out += [model_out['last_hidden_state'].mean(1)]
            # second_last_hidden_out = model_out['hidden_states'][-2].mean(1)
            out_for_on_text += [model_out['pooler_output']]
        if last_hidden_out == []:
            continue
        last_hidden_out = torch.cat(last_hidden_out, dim=0)
        out_for_on_text = torch.cat(out_for_on_text, dim=0)
        num_sent, ptm_size = out_for_on_text.shape
        for key, emb in {'last_hidden_out': last_hidden_out, 'pooler_output': out_for_on_text}.items():
            if args.max_sent > num_sent:
                emb = torch.cat([emb, torch.zeros(args.max_sent - num_sent, ptm_size).cuda()], dim=0)
            else:
                emb = emb[:args.max_sent, :]
            output_dct[data_dir][key] = emb.cpu().detach().numpy()
    
    pickle.dump(output_dct, open(args.save_path, 'wb'))

def generate_embedding_with_cuda(args, tokenizer, model):
    
    data_list = os.listdir(args.data_path)
    output_dct = {}
    for data_dir in tqdm(data_list):
        output_dct[data_dir] = {}
        text_path = args.data_path + data_dir + '/Text.txt'
        text_file = open(text_path)
        last_hidden_out, second_last_hidden_out, out_for_on_text = [], [], []
        for line in text_file.readlines():
            line = line.strip()
            # all_sent_for_one_text += [line]
            try:
                input = tokenizer(line, padding=True, return_tensors='pt')
            except:
                print('wrong tokenize, sent_list_from: {}'.format(data_dir))
                continue
            model_out = model(input['input_ids'].cuda(), input['token_type_ids'].cuda(), input['attention_mask'].cuda())
        
            last_hidden_out += [model_out['last_hidden_state'].mean(1).cpu()]
            second_last_hidden_out += [model_out['hidden_states'][-2].mean(1).cpu()]
            out_for_on_text += [model_out['pooler_output'].cpu()]
        last_hidden_out = torch.cat(last_hidden_out, dim=0)
        second_last_hidden_out = torch.cat(second_last_hidden_out, dim=0)
        out_for_on_text = torch.cat(out_for_on_text, dim=0)
        num_sent, ptm_size = out_for_on_text.shape
        for key, emb in {'last_hidden_out': last_hidden_out, 'second_last_hidden_out': second_last_hidden_out, 'pooler_output': out_for_on_text}.items():
            if args.max_sent > num_sent:
                emb = torch.cat([emb, torch.zeros(args.max_sent - num_sent, ptm_size)], dim=0)
            else:
                emb = emb[:args.max_sent, :]
            output_dct[data_dir][key] = emb.detach().numpy()
    
    pickle.dump(output_dct, open(args.save_path_with_cuda, 'wb'))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='get embedding')
    
    parser.add_argument('--ptm_type', default='/your/project/path/pretrain_ckpt/', type=str)
    parser.add_argument('--data_path', default='/your/project/path/raw_data/ReleasedDataset_mp3/', type=str)     # /your/project/path/raw_data/ReleasedDataset_mp3/
    parser.add_argument('--max_sent', default=512, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--save_path', default='/your/project/path/text_embedding/kept.pkl', type=str)
    parser.add_argument('--save_path_with_cuda', default='/your/project/path/pklFiles/kept_cuda.pkl', type=str)
    args = parser.parse_args()

    print(args)

    data_dir = '/your/dataset/path'
    new_tokens = []
    with open(data_dir + 'data_process/needed_files/final_wiki_ids.json', 'r') as f:
        wiki_ids = json.load(f)
    for line in wiki_ids.keys():
        new_tokens += [line.strip()]
    
    tokenizer = AutoTokenizer.from_pretrained(args.ptm_type)  # args.ptm_type
    args.device = 'cuda'
    
    ptm = AutoModel.from_pretrained(args.ptm_type).to(args.device)   # .to(args.device)
    
    generate_embedding(args, tokenizer, ptm)
    # generate_embedding_with_cuda(args, tokenizer, model)
