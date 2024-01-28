import os
import numpy as np
import sys
import argparse
import pandas as pd
import logging
from time import strftime, localtime
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import math
from data_utils import set_seed
from time_models.CondAutoformer import CondAutoformer
from time_models.CondTransformer import CondTransformer
from data_utils import DictDataset, select_dataset
from torch.utils.data import DataLoader, RandomSampler
from time_models import Autoformer, Transformer
from transformers_model.modules import MergeFeature


def ModifyData(args, df,single_df, price_df, raw_data_path):
    data_dict = {}

    data_dict['X_past']=[]
    data_dict['X_graph']=[]
    
    data_dict['X_audio'] = []
    
    data_dict['y_avg3days']=[]
    data_dict['y_avg7days']=[]
    data_dict['y_avg15days']=[]
    data_dict['y_avg30days']=[]
    
    data_dict['y_single3days']=[]
    data_dict['y_single7days']=[]
    data_dict['y_single15days']=[]
    data_dict['y_single30days']=[]
    
    data_dict['y_price3days']=[]
    data_dict['y_price7days']=[]
    data_dict['y_price15days']=[]
    data_dict['y_price30days']=[]

    error_audio=[]

    call_dict = {}
    call_single_dict = {}

    for index,row in tqdm(df.iterrows()):

        if row['text_file_name'] not in call_dict.keys():
            call_dict[row['text_file_name']] = index
        else:
            continue

        empty_graph_embd= np.zeros((200), dtype=np.float64)
        empty_graph_embd_3days= np.zeros((512, args.text_indim), dtype=np.float64)
            
        if args.dataset == 'ec':
            lstm_matrix_temp = np.zeros((512, args.text_indim), dtype=np.float64)
            i=0
            try:
                audio_path = raw_data_path + row['text_file_name'] + '/audio_from_hubert_base_superb_ks.pkl'
                with open(audio_path, "rb") as f:
                    audio_matrix = pickle.load(f)
                
                audio_matrix = np.array(audio_matrix).squeeze(1)
                length = audio_matrix.shape[0]
                if length < 512:
                    audio_matrix = np.concatenate((audio_matrix, np.zeros((512-length, args.text_indim), dtype=np.float64)), axis=0)
                else:
                    audio_matrix = audio_matrix[:512, :]
                data_dict['X_audio'].append(audio_matrix)
            except:
                data_dict['X_audio'].append(lstm_matrix_temp)
                error_audio.append(row['text_file_name'])
                print(row['text_file_name'])
            
            data_dict['X_past'].append(list(row[35:]))
            
        else:    # maec
            lstm_matrix_temp = np.zeros((512, 29), dtype=np.float64)
            i=0
            try:
                audio_path = pd.read_csv(raw_data_path + row['text_file_name'] + '/features.csv')
                dtype_dict = dict(audio_path.dtypes)
                for k in dtype_dict.keys():
                    if dtype_dict[k] == 'object':
                        audio_path.loc[:, k] = pd.to_numeric(audio_path.loc[:, k], errors='coerce').fillna(0)
                
                audio_matrix = audio_path.values
                length = audio_matrix.shape[0]
                if length < 512:
                    audio_matrix = np.concatenate((audio_matrix, np.zeros((512-length, 29), dtype=np.float64)), axis=0)
                else:
                    audio_matrix = audio_matrix[:512, :]
                data_dict['X_audio'].append(audio_matrix)
            except:
                data_dict['X_audio'].append(lstm_matrix_temp)
                error_audio.append(row['text_file_name'])
                print(row['text_file_name'])
            
            data_dict['X_past'].append(np.array(pd.to_numeric(row[['past_{}'.format(31-i) for i in range(2,31)]], errors='coerce').fillna(0).values, dtype=np.float64))
            
        
        try:
            # data_dict['X_graph3'].append(graph_embd_dict[row['text_file_name']])
            data_dict['X_graph'].append(graph_embd_dict[row['text_file_name']]['pooler_output'])
        except KeyError:
            data_dict['X_graph'].append(empty_graph_embd_3days)
        
        #Past 30 Days avg vol
        # data_dict['X_past'].append(list(row[35:]))
        # data_dict['X_past'].append(np.array(row[['past_{}'.format(31-i) for i in range(1,30)]].values, dtype=np.float32))
        
        data_dict['y_avg3days'].append(float(row['future_3']))
        data_dict['y_avg7days'].append(float(row['future_7']))
        data_dict['y_avg15days'].append(float(row['future_15']))
        data_dict['y_avg30days'].append(float(row['future_30']))
        
        data_dict['y_price3days'].append(float(price_df[price_df.text_file_name == row['text_file_name']]['future_label_3']))
        data_dict['y_price7days'].append(float(price_df[price_df.text_file_name == row['text_file_name']]['future_label_7']))
        data_dict['y_price15days'].append(float(price_df[price_df.text_file_name == row['text_file_name']]['future_label_15']))
        data_dict['y_price30days'].append(float(price_df[price_df.text_file_name == row['text_file_name']]['future_label_30']))
        
    for index,row in single_df.iterrows():

        if row['text_file_name'] not in call_single_dict.keys():
            call_single_dict[row['text_file_name']] = index
        else:
            continue

        data_dict['y_single3days'].append(float(row['future_Single_3']))
        data_dict['y_single7days'].append(float(row['future_Single_7']))
        data_dict['y_single15days'].append(float(row['future_Single_15']))
        data_dict['y_single30days'].append(float(row['future_Single_30']))


    # X_past = seasonal_price['price_feature']    
    data_dict['X_past']=np.array(data_dict['X_past'])
    
    data_dict['X_graph']=np.nan_to_num(np.array(data_dict['X_graph']))
    
    data_dict['X_audio']=np.nan_to_num(np.array(data_dict['X_audio']))
    
    data_dict['y_avg3days']=np.array(data_dict['y_avg3days'])
    data_dict['y_avg7days']=np.array(data_dict['y_avg7days'])
    data_dict['y_avg15days']=np.array(data_dict['y_avg15days'])
    data_dict['y_avg30days']=np.array(data_dict['y_avg30days'])
    
    data_dict['y_single3days']=np.array(data_dict['y_single3days'])
    data_dict['y_single7days']=np.array(data_dict['y_single7days'])
    data_dict['y_single15days']=np.array(data_dict['y_single15days'])
    data_dict['y_single30days']=np.array(data_dict['y_single30days'])
    
    data_dict['y_price3days']=np.array(data_dict['y_price3days'])
    data_dict['y_price7days']=np.array(data_dict['y_price7days'])
    data_dict['y_price15days']=np.array(data_dict['y_price15days'])
    data_dict['y_price30days']=np.array(data_dict['y_price30days'])
    
    data_dict['y_avg3days'][np.isinf(data_dict['y_avg3days'])] = 0.0
    data_dict['y_avg7days'][np.isinf(data_dict['y_avg7days'])] = 0.0
    data_dict['y_avg15days'][np.isinf(data_dict['y_avg15days'])] = 0.0
    data_dict['y_avg30days'][np.isinf(data_dict['y_avg30days'])] = 0.0
    
    data_dict['y_single3days'][np.isinf(data_dict['y_single3days'])] = 0.0
    data_dict['y_single7days'][np.isinf(data_dict['y_single7days'])] = 0.0
    data_dict['y_single15days'][np.isinf(data_dict['y_single15days'])] = 0.0
    data_dict['y_single30days'][np.isinf(data_dict['y_single30days'])] = 0.0
    
    data_dict['y_price3days'][np.isinf(data_dict['y_price3days'])] = 0.0
    data_dict['y_price7days'][np.isinf(data_dict['y_price7days'])] = 0.0
    data_dict['y_price15days'][np.isinf(data_dict['y_price15days'])] = 0.0
    data_dict['y_price30days'][np.isinf(data_dict['y_price30days'])] = 0.0
    
    data_dict['y_avg3days']=np.nan_to_num(data_dict['y_avg3days'])
    data_dict['y_avg7days']=np.nan_to_num(data_dict['y_avg7days'])
    data_dict['y_avg15days']=np.nan_to_num(data_dict['y_avg15days'])
    data_dict['y_avg30days']=np.nan_to_num(data_dict['y_avg30days'])
    
    data_dict['y_single3days']=np.nan_to_num(data_dict['y_single3days'])
    data_dict['y_single7days']=np.nan_to_num(data_dict['y_single7days'])
    data_dict['y_single15days']=np.nan_to_num(data_dict['y_single15days'])
    data_dict['y_single30days']=np.nan_to_num(data_dict['y_single30days'])
    
    data_dict['y_price3days']=np.nan_to_num(data_dict['y_price3days'])
    data_dict['y_price7days']=np.nan_to_num(data_dict['y_price7days'])
    data_dict['y_price15days']=np.nan_to_num(data_dict['y_price15days'])
    data_dict['y_price30days']=np.nan_to_num(data_dict['y_price30days'])
    
#     args.logger.info(np.sum(y_3days))
#     args.logger.info(np.sum(y_7days))
#     args.logger.info(np.sum(y_15days))
#     args.logger.info(np.sum(y_30days))
    
    data_dict['X_past']=np.nan_to_num(data_dict['X_past'])
    
    n_features = 1
    data_dict['X_past'] = data_dict['X_past'].reshape((data_dict['X_past'].shape[0], data_dict['X_past'].shape[1], n_features))

    for k in data_dict.keys():
        data_dict[k] = torch.tensor(data_dict[k]).float()
        
    return DictDataset(**data_dict)


class CondInfer(nn.Module):
    def __init__(self, args):
        super(CondInfer, self).__init__()
        time_model_dict = {
                'CondTransformer': CondTransformer,
                'CondAutoformer': CondAutoformer,
            }
        self.model = time_model_dict[args.time_model](args)   # CondAutoformer
        self.args = args
        
        
        hidden_size = args.c_out
        self.avg_pred = nn.Linear(hidden_size, 1)
        self.single_pred = nn.Linear(hidden_size, 1)
        
        self.fuse = MergeFeature(text_in_dim=768, audio_in_dim=args.audio_indim, h_dim=200, out_dim=200)
        
        self.classifier_avg = nn.Linear(hidden_size, 2)
    
    def forward(self, x, cond, X_audio):
        # cond = self.fuse(cond, X_audio)
        out = self.model(x, cond)
        avg_out = self.avg_pred(out)
        single_out = self.single_pred(out)
        avg_cls = self.classifier_avg(out)
        return avg_out.squeeze(-1), single_out.squeeze(-1), avg_cls

def select_inputs(args, batch):

    inbatch = {k: v.to(args.device) for k, v in batch.items()}
    
    X_audio = inbatch['X_audio']
    X_past = inbatch['X_past']
    X_graph = inbatch['X_graph']
    y_avg = inbatch['y_avg{}days'.format(str(args.duration))]
    y_single = inbatch['y_single{}days'.format(str(args.duration))]
    y_price = inbatch['y_price{}days'.format(str(args.duration))]

    return X_past, X_graph, y_avg, y_single, y_price, X_audio

def eval(args, model, dataloader, evaluation, e, best_mse, best_f1, eval_type='dev'):
    
    with torch.no_grad():
        
        model.eval()

        loss_avg = 0.0
        loss_single = 0.0
        pred = []
        all_out, all_labels, pred = [], [], []
        for i, batch in enumerate(dataloader):

            X_past, X_graph, y_avg, y_single, y_price, X_audio = select_inputs(args, batch)
            
            if args.time_model_to_save:
                avg_pred, single_pred, _ = model(X_past, X_graph)
            else:
                avg_pred, single_pred, avg_cls = model(X_past, X_graph, X_audio)
            if args.run_mode == 'reg':
                loss_avg += F.mse_loss(avg_pred, y_avg)
                loss_single += F.mse_loss(single_pred, y_single)
                pred += avg_pred.cpu().tolist()
            elif args.run_mode == 'cls':
                all_out += [np.argmax(avg_cls.float().cpu().numpy(), axis=1)]
                all_labels += [y_price.long().cpu().numpy()]
                pred += avg_cls.cpu().tolist()
            
        
        loss_avg = loss_avg/(i+1)
        loss_single = loss_single/(i+1)
        if args.run_mode == 'reg': 
            evaluation['Epoch'].append(e)
            if eval_type == 'dev':
                args.logger.info('Epoch: {}, Dev MSE: {}, Dev MSE AUXILIARY: {}'.format(e, loss_avg.item(), loss_single.item()))
                evaluation['Dev MSE'].append(loss_avg.item())
                evaluation['Dev MSE AUXILIARY'].append(loss_single.item())
            
            elif eval_type == 'test':
                args.logger.info('Epoch: {}, Test MSE: {}, Test MSE AUXILIARY: {} \n'.format(e, loss_avg.item(), loss_single.item()))
                evaluation['Test MSE'].append(loss_avg.item())
                evaluation['Test MSE AUXILIARY'].append(loss_single.item())
                pred_df = pd.DataFrame(pred)
                pred_df.to_csv(os.path.join('/your/project/path/preds_dir/', args.pred_save_dir+'/'+args.run_mode +'/final_serise_infer_pred_'+str(args.duration)+'.csv'))
            
            if eval_type == 'dev' and e != 'final':
                if best_mse > loss_avg.item():
                    print('save model ..., best_dev_mse: {}'.format(str(best_mse)))
                    save_path = '/your/project/path/{}/{}/{}/final_serise_infer_best_pkls_{}_{}_dur{}_{}.pth.tar'.format(args.pkl_save_path, args.dataset, args.duration, args.dataset, args.pid, args.duration, args.time_model)
                    save_path_dir = '/'.join(save_path.split('/')[:-1])
                    if not os.path.exists(save_path_dir):
                        os.makedirs(save_path_dir)
                    torch.save(model.state_dict(), save_path)
                    best_mse = loss_avg.item()
        
        elif args.run_mode == 'cls':
            all_labels = np.concatenate(all_labels)
            all_out = np.concatenate(all_out)
            evaluation['Epoch'].append(e)
            if eval_type == 'dev':
                f1 = f1_score(all_labels, all_out)
                acc = accuracy_score(all_labels, all_out)
                matt = matthews_corrcoef(all_labels, all_out)
                args.logger.info('Epoch: {}, Dev F1: {}, Dev ACC: {}, Dev MATT: {}'.format(e, f1, acc, matt))
                # evaluation['Dev MSE'].append(loss_test.item())
                # evaluation['Dev MSE AUXILIARY'].append(loss_test_b.item())
                evaluation['Dev F1'].append(f1)
                evaluation['Dev ACC'].append(acc)
                evaluation['Dev MATT'].append(matt)
            
            elif eval_type == 'test':
                f1 = f1_score(all_labels, all_out)
                acc = accuracy_score(all_labels, all_out)
                matt = matthews_corrcoef(all_labels, all_out)
                args.logger.info('Epoch: {}, Test F1: {}, Test ACC: {}, Test MATT: {} \n'.format(e, f1, acc, matt))
                # evaluation['Test MSE'].append(loss_test.item())
                # evaluation['Test MSE AUXILIARY'].append(loss_test_b.item())
                evaluation['Test F1'].append(f1)
                evaluation['Test ACC'].append(acc)
                evaluation['Test MATT'].append(matt)
                
            if eval_type == 'dev' and e != 'final':
                if best_f1 < f1:
                    save_path = '/your/project/path/{}/{}/{}/cls_final_serise_infer_best_pkls_{}_{}_dur{}_{}.pth.tar'.format(args.pkl_save_path, args.dataset, args.duration, args.dataset, args.pid, args.duration, args.time_model)
                    save_path_dir = '/'.join(save_path.split('/')[:-1])
                    if not os.path.exists(save_path_dir):
                        os.makedirs(save_path_dir)
                    torch.save(model.state_dict(), save_path)
                    best_f1 = f1.item()   
    
    return evaluation, best_mse, loss_avg, loss_single, best_f1

def save_embedding(args, model, train_dataloader, test_dataloader, dev_dataloader):
    
    with torch.no_grad():
        
        model.eval()
        datas = {'train': train_dataloader, 'test': test_dataloader, 'dev': dev_dataloader}
        for name, data in datas.items():
            out_dict = []
            loss_avg = 0.0
            loss_single = 0.0
            pred = []
            for i, batch in enumerate(data):

                X_past, X_graph, y_avg, y_single, y_price = select_inputs(args, batch)
                
                avg_pred, single_pred, price_hidden = model(X_past, X_graph)
                loss_avg += F.mse_loss(avg_pred, y_avg)
                loss_single += F.mse_loss(single_pred, y_single)
                pred += avg_pred.cpu().tolist()

                out_dict += [price_hidden.detach().cpu().numpy()]
        
            pickle.dump(out_dict, open(args.saved_feature_dir + '/' + args.source + '/feaures_' + name + '_' + str(args.duration) + 'days_' + args.source + '_' + args.run_mode + '.pkl', 'wb'))
    

def runner(args, train_dataset, test_dataset, dev_dataset, avg_day_mse_list, single_day_mse_list):

    trainloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
    testloader = DataLoader(test_dataset, sampler=RandomSampler(test_dataset), batch_size=args.batch_size)
    devloader = DataLoader(dev_dataset, sampler=RandomSampler(dev_dataset), batch_size=args.batch_size)
    
    if args.time_model_to_save:
        model = TimeInfer(args).to(args.device)
    else:
        model = CondInfer(args).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), amsgrad=False, weight_decay=args.weight_decay)   #weight_decay
    
    criterion = nn.MSELoss()
    
    best_mse = 20
    best_f1 = 0.0
    evaluation= {'Epoch': [],'Train MSE': [], 'Dev MSE':[], 'Dev MSE AUXILIARY':[], 'Test MSE':[], 'Test MSE AUXILIARY':[], 'Dev F1': [], \
                        'Dev ACC': [], 'Dev MATT': [], 'Test F1': [], 'Test ACC': [], 'Test MATT': []}
    
    step = 0
    
    for e in tqdm(range(args.epochs)):
        train_loss_tol = 0.0
        model.train()
        for i, batch in enumerate(trainloader):
            
            step += 1
            if  step % 100 == 0:
                args.gumbel_temprature = max( np.exp((step+1) *-1* args.gumbel_decay), .05)

            opt.zero_grad()

            X_past, X_graph, y_avg, y_single, y_price, X_audio = select_inputs(args, batch)
            if args.time_model_to_save:
                avg_pred, single_pred, _ = model(X_past, X_graph,)
            else:
                avg_pred, single_pred, avg_cls = model(X_past, X_graph, X_audio)
            avg_loss = criterion(avg_pred, y_avg)
            single_loss = criterion(single_pred, y_single)
            if args.run_mode == 'reg':
                loss = args.mu * avg_loss + (1-args.mu) * single_loss
                # loss = avg_loss
            elif args.run_mode == 'cls':
                loss = nn.CrossEntropyLoss()(avg_cls, y_price.long())
            train_loss_tol += loss

            loss.backward()

            if args.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            
            opt.step()

        train_loss_tol = train_loss_tol / (i+1)
        args.logger.info('Epoch: {}, Train loss: {}'.format(e, train_loss_tol))
        evaluation['Train MSE'].append(train_loss_tol.item())

        evaluation, best_mse, _, _, best_f1 = eval(args, model, devloader, evaluation, e, best_mse, best_f1, eval_type='dev')
        evaluation, _, _, _, _ = eval(args, model, testloader, evaluation, 'test', best_mse, best_f1, eval_type='test')

    model.load_state_dict(torch.load('/your/project/path/{}/{}/{}/final_serise_infer_best_pkls_{}_{}_dur{}_{}.pth.tar'.format(args.pkl_save_path, args.dataset, args.duration, args.dataset, args.pid, args.duration, args.time_model)))          
    evaluation, best_mse, loss_avg, loss_single, _ = eval(args, model, testloader, evaluation, 'test', best_mse, best_f1, eval_type='test')
    avg_day_mse_list.append(loss_avg.item())
    single_day_mse_list.append(loss_single.item())
    if args.time_model_to_save:
        train_dataloader_forsave = DataLoader(train_dataset, batch_size=1)
        test_dataloader_forsave = DataLoader(test_dataset, batch_size=1)
        dev_dataloader_forsave = DataLoader(dev_dataset, batch_size=1)
        print('save embedding ...')
        save_embedding(args, model, train_dataloader_forsave, test_dataloader_forsave, dev_dataloader_forsave)

    evaluation, best_mse, _, _, best_f1 = eval(args, model, devloader, evaluation, 'final', best_mse, best_f1, eval_type='dev')

    return avg_day_mse_list, single_day_mse_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0002, type=float)  # 0.0005
    parser.add_argument('--epochs', default=200, type=int)  # 100
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--kernel_size', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--duration', default=30, type=int)
    parser.add_argument('--mu', default=0.7, type=float)
    parser.add_argument('--gradient_clipping', default=0.0, type=float)  # 1.0
    parser.add_argument('--pid', default='6', type=str)
    parser.add_argument('--pred_save_dir', default='text_dir', type=str)
    parser.add_argument('--run_mode', default='reg', type=str)  #cls, reg
    parser.add_argument('--raw_data_path', default='/your/project/path/MAECdata/MAEC_Dataset/', type=str)  # /your/project/path/MAECdata/MAEC_Dataset/    /your/project/path/raw_data/ReleasedDataset_mp3/
    parser.add_argument('--dist_threshold', default=0.95, type=float)
    parser.add_argument('--dataset', default='ec', type=str, choices=['ec', '15', '16'])
    parser.add_argument('--audio_indim', default=29, type=int) # 768 for ec
    parser.add_argument('--text_indim', default=768, type=int) # 768 for ec
    parser.add_argument('--text_embedding', type=str, default='raw_bert_base_uncased') # bert-base : raw_bert_base_uncased; prosuai finbert: raw_prosusai_finbert; knowledge MLM : bert_base_uncased_pretrain_descrip_ep60; raw MLM: bert_base_uncased_pretrain_raw_ep60
    # flang-bert: ec_embed_flang_bert_raw
    parser.add_argument('--pkl_save_path', default='save_pkls', type=str)
    parser.add_argument('--log_save_path', default='temp_log', type=str)

    # time model
    parser.add_argument('--gumbel_temprature', default=1, type=int)
    parser.add_argument('--gumbel_decay', default=1e-05, type=float)
    parser.add_argument('--time_model_to_save', default=False, type=bool)
    parser.add_argument('--time_model', type=str, default='Autoformer', choices=['Autoformer', 'Transformer', 'Informer', \
                                                                                      'CondAutoformer', 'CondTransformer', 'CondFEDformer', 'CondInformer'])
    parser.add_argument('--saved_feature_dir', default='/your/project/path/save_features', type=str)
    parser.add_argument('--source', default='price_reviseautofm', type=str)  # text&audio&price
    parser.add_argument('--freq', default='h', type=str)

    # condautoformer
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--moving_avg', type=int, default=5, help='window size of moving average')
    parser.add_argument('--enc_in', default=1, type=int)
    parser.add_argument('--c_out', default=50, type=int)
    parser.add_argument('--d_model', default=200, type=int, help='hidden size')  # 100
    parser.add_argument('--embed', default='timeF', type=str)
    parser.add_argument('--embed_type', default='timeF', type=str)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=400, help='dimension of fcn')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--e_layers', default=2, type=int)
    parser.add_argument('--d_layers', default=1, type=int)

    parser.add_argument('--seq_len', default=29, type=int, help='input series length')
    parser.add_argument('--mode_select', default='random', help='for fedformer')
    parser.add_argument('--modes', default=32, type=int, help='for fedformer')
    
    args = parser.parse_args()
    
    logger = logging.getLogger('cli')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    log_file = '{}-{}-{}-{}-{}-{}.log'.format('final_series_infer', strftime("%Y-%m-%d_%H:%M:%S", localtime()), args.dataset, args.duration, args.dist_threshold, args.time_model)
    logger.addHandler(logging.FileHandler("%s/%s" % ('/your/project/path/log/{}'.format(args.log_save_path), log_file)))
    args.logger = logger

    args.logger.info(args)

    out_path = '/your/project/path/output/'
    base_dir = '/your/dataset/path/'
    feature_dir = '/your/project/path/save_features'
    
    set_seed(args.seed)
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    traindf,single_traindf, price_df_train, testdf,single_testdf, price_df_test, valdf,single_valdf, price_df_val, graph_embd_dict \
                    = select_dataset(args, base_dir)
    
    train_dataset = ModifyData(args, traindf,single_traindf, price_df_train, args.raw_data_path)

    test_dataset = ModifyData(args, testdf,single_testdf, price_df_test, args.raw_data_path)

    dev_dataset = ModifyData(args, valdf,single_valdf, price_df_val, args.raw_data_path)

    avg_day_mse_list,  single_day_mse_list = [], []

    for i in range(10):
        args.logger.info("i="+str(i))
        avg_day_mse_list, single_day_mse_list = runner(args, train_dataset, test_dataset, dev_dataset, avg_day_mse_list, single_day_mse_list)
        args.logger.info('duration: {}'.format(str(args.duration)))
        args.logger.info("Avg MEAN MSE: {}".format(np.array(avg_day_mse_list).mean()))
        args.logger.info("SINGLE MEAN MSE: {}".format(np.array(single_day_mse_list).mean()))
        args.logger.info("Avg Standard Deviation MSE: {}".format(np.std(np.array(avg_day_mse_list))))
        args.logger.info("SINGLE Standard Deviation MSE: {}".format(np.std(np.array(single_day_mse_list))))
        args.logger.info('Avg MSE: {}'.format(avg_day_mse_list))
        args.logger.info('SINGLE MSE: {}'.format(single_day_mse_list))

    avg_day_mse_df=pd.DataFrame(avg_day_mse_list)
    avg_day_mse_df.to_csv(out_path + '3GCN_LSTM_boxplot_cond_avg_day_mse_df.csv')

    single_day_mse_df=pd.DataFrame(single_day_mse_list)
    single_day_mse_df.to_csv(out_path + '3GCN_LSTM_boxplot_cond_single_day_mse_df.csv')

