import os
import numpy as np
import torch
import pandas as pd
from abc import ABC, abstractmethod
import pickle
from typing import List
from torch.utils.data import Dataset
import torch.nn as nn
import random
from tqdm import tqdm



class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()

        assert all(next(iter(tensors.values())).size(0) == tensor.size(0)
                   for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)



def select_dataset(args, base_dir):
    
    if args.dataset == 'ec':
        
        with open(base_dir + 'text_embedding/{}.pkl'.format(args.text_embedding), 'rb') as f:   # 
            graph_embd_dict=pickle.load(f)
            
        traindf= pd.read_csv(base_dir + "price_data/train_split_Avg_Series_WITH_LOG.csv")
        testdf=pd.read_csv(base_dir + "price_data/test_split_Avg_Series_WITH_LOG.csv")
        valdf=pd.read_csv(base_dir + "price_data/val_split_Avg_Series_WITH_LOG.csv")

        single_traindf=pd.read_csv(base_dir + "price_data/train_split_SeriesSingleDayVol3.csv")
        single_testdf=pd.read_csv(base_dir + "price_data/test_split_SeriesSingleDayVol3.csv")
        single_valdf=pd.read_csv(base_dir + "price_data/val_split_SeriesSingleDayVol3.csv")
        
        price_df_train = pd.read_csv(os.path.join(base_dir, 'price_data/train_price_label.csv'))
        price_df_test = pd.read_csv(os.path.join(base_dir, 'price_data/test_price_label.csv'))
        price_df_val = pd.read_csv(os.path.join(base_dir, 'price_data/dev_price_label.csv'))
    
    else:
        
        with open(base_dir + 'text_embedding/{}.pkl'.format(args.text_embedding), 'rb') as f:   #  bert_base_uncased_pretrain_descrip_ep60_maec   finbert_earnings_from_mypretrain_with_kg_ep14.pkl   
            graph_embd_dict=pickle.load(f)
        
        traindf= pd.read_csv(base_dir + "price_data/maec/{}/maec{}_train_avg_val.csv".format(args.dataset, args.dataset))    # avg_df
        testdf=pd.read_csv(base_dir + "price_data/maec/{}/maec{}_test_avg_val.csv".format(args.dataset, args.dataset))
        valdf=pd.read_csv(base_dir + "price_data/maec/{}/maec{}_dev_avg_val.csv".format(args.dataset, args.dataset))

        single_traindf=pd.read_csv(base_dir + "price_data/maec/{}/maec{}_train_single_val.csv".format(args.dataset, args.dataset))   #single_df
        single_testdf=pd.read_csv(base_dir + "price_data/maec/{}/maec{}_test_single_val.csv".format(args.dataset, args.dataset))
        single_valdf=pd.read_csv(base_dir + "price_data/maec/{}/maec{}_dev_single_val.csv".format(args.dataset, args.dataset))
        
        price_df_train = pd.read_csv(os.path.join(base_dir, "price_data/maec/{}/maec{}_train_price_label.csv".format(args.dataset, args.dataset)))  # price_df    
        price_df_test = pd.read_csv(os.path.join(base_dir, "price_data/maec/{}/maec{}_test_price_label.csv".format(args.dataset, args.dataset))) 
        price_df_val = pd.read_csv(os.path.join(base_dir, "price_data/maec/{}/maec{}_dev_price_label.csv".format(args.dataset, args.dataset)))
        
        
    return traindf,single_traindf, price_df_train, testdf,single_testdf, price_df_test, valdf,single_valdf, price_df_val, graph_embd_dict