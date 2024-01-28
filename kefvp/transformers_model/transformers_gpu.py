import os
import time
import matplotlib
from tqdm import tqdm_notebook
from datetime import date
from matplotlib import pyplot as plt
from pylab import rcParams

#math package
import random, math
from numpy.random import seed
# from tensorflow import set_random_seed
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

#Pytorch Package
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForPreTraining

#Customized Transformers Util
# from transformer import util
from .util import mask_, contains_nan
from .util import d
from latent.kumadist import IndependentLatentModel, DependentLatentModel
from transformers_model.modules import CrossAttention, GraphConvolution, CrossFusion
from time_models import Informer, Autoformer, Transformer, ReviseAutoformer

class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities

        assert not contains_nan(dot[:, 1:, :]) # only the forst row may contain nan

        if self.mask == 'first':
            dot = dot.clone()
            dot[:, :1, :] = 0.0
            # - The first row of the first attention matrix is entirely masked out, so the softmax operation results
            #   in a division by zero. We set this row to zero by hand to get rid of the NaNs

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x
##RTransformer

class KUMARTransformer(nn.Module):
    """
    Transformer for sequences Regression    
    
    """

    def __init__(self, args, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0):
        """
        emb: Embedding dimension
        heads: nr. of attention heads
        depth: Number of transformer blocks
        seq_length: Expected maximum sequence length
        num_tokens: Number of tokens (usually words) in the vocabulary
        num_classes: Number of classes.
        max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()
        self.args = args
        self.source_list = self.args.source.split('_')
        
        self.num_tokens, self.max_pool = num_tokens, max_pool

        #self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)
        
        if 'text' in self.source_list:
            self.ptm = AutoModel.from_pretrained(self.args.ptm)
            self.trans_bert = nn.Linear(emb, args.audio_embed_size)
            self.gcn = GraphConvolution(emb, args.audio_embed_size)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=args.audio_embed_size, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)
        
        ######Audio######
        if 'audio' in self.source_list:
            self.audio_linear = nn.Linear(self.args.audio_embed, args.audio_embed_size)
            self.audio_model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=args.audio_embed_size, nhead=self.args.audio_hd),
                num_layers=self.args.num_audio_layers, 
                norm=nn.LayerNorm(args.audio_embed_size)
            )
        
        #########Time Series####
        if 'price' in self.source_list:
            time_model_dict = {
                'Autoformer': Autoformer,
                'Transformer': Transformer,
                'Informer': Informer,
                'ReviseAutoformer': ReviseAutoformer
            }
            self.time_model = time_model_dict[self.args.time_model].Model(self.args).float()
            
        ####Cross Attention#####
        # self.cross_att = CrossAttention(text_in=args.embedding_size, audio_in=args.audio_embed, \
        #     h_dim=args.corss_hidden, head=args.cross_head)
        if 'audio' in self.source_list and 'text' in self.source_list:
            self.modal_fusion = CrossFusion(text_indim=args.audio_embed_size, audio_indim=args.audio_embed_size, h_dim=args.corss_hidden)
        
        ####KUMA####
        # if self.args.dependent_z:
        #     self.latent_model = DependentLatentModel(
        #         embed_size=args.audio_embed_size, hidden_size=args.kuma_hidden_size,
        #         dropout=args.kuma_dropout, layer='lstm')
        # else:
        #     self.latent_model = IndependentLatentModel(
        #         embed_size=args.audio_embed_size, hidden_size=args.kuma_hidden_size,
        #         dropout=args.kuma_dropout, layer='lstm')
        
        # emb = args.corss_hidden * 2
        emb = args.audio_embed_size
        
        self.toprobs = nn.Linear(emb, num_classes)
        self.toprobs_b = nn.Linear(emb, num_classes)
        self.do = nn.Dropout(dropout)

    def forward(self, x, ratio_text_ids, ratio_type_ids, ratio_attention_mask, audio_x, price_x, ratio_pooler, audio_embedding, X_graph3):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        sentences_emb = x
        x_mask = (sentences_emb[:, :, 0] != 0)
        # x_mask = (sentences_emb != 0)
        b, t, e = x.size()
        # x_sentence_mask = (ratio_text_ids[:, :, 0] != 0)
        # adj = adj.view(-1, self.args.ratio_max_length, self.args.ratio_max_length)
        
        if 'text' in self.source_list:
            ratio_out = self.ptm(ratio_text_ids.view(-1, self.args.ratio_max_length), ratio_type_ids.view(-1, self.args.ratio_max_length), ratio_attention_mask.view(-1, self.args.ratio_max_length))   # 72 *2  *  768
            ratio_pooler, ratio_hidden = ratio_out['pooler_output'], ratio_out['last_hidden_state']
            ratio_pooler = ratio_pooler.view(b, -1, e)
            ratio_pooler = F.dropout(ratio_pooler, p=0.5)
            text_hidden = self.trans_bert(ratio_pooler)
            x = text_hidden.mean(1)
            
            # ratio_hidden = self.gcn(ratio_hidden, adj)
            # text_hidden = self.trans_bert(ratio_pooler) + (ratio_hidden.view(b, -1, self.args.ratio_max_length, self.args.audio_embed_size)*aspect_pos_mask.unsqueeze(-1)).mean(2)
            
            # x = self.trans_bert(ratio_pooler).mean(1) + (ratio_hidden.view(b, -1, self.args.ratio_max_length, self.args.audio_embed_size)*aspect_pos_mask.unsqueeze(-1)).mean(1).mean(1)
            # x = torch.cat([self.trans_bert(ratio_pooler).mean(1), (ratio_hidden.view(b, -1, self.args.ratio_max_length, self.args.audio_embed_size)*aspect_pos_mask.unsqueeze(-1)).mean(1).mean(1)], dim=-1)
            
        if 'price' in self.source_list:
            seasonal_price_hidden, trend_price_hidden = self.time_model(price_x, price_x)
            price_hidden = F.dropout(trend_price_hidden)
            x = price_hidden.mean(1)

            # dec_out, trend_hidden = self.time_model(price_x)
            # x = F.dropout(dec_out)
            
        
        # positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        # x = sentences_emb.cuda() + positions
        
        # x = self.do(x)
        
        # x = self.tblocks(x)
        
        # x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension
        
        if 'audio' in self.source_list:
            audio_hidden = self.audio_model(self.audio_linear(audio_embedding))
            audio_hidden = F.dropout(audio_hidden)
            x = audio_hidden.mean(1)
        
        if 'audio' in self.source_list and 'text' in self.source_list and 'transfuse' not in self.source_list:
            x = self.modal_fusion(text_hidden, audio_hidden)
        
        if 'audio' in self.source_list and 'text' in self.source_list and 'transfuse' in self.source_list:
            fuse_emb = torch.cat([text_hidden, audio_hidden], dim=-2)
            x = self.tblocks(fuse_emb)
            x = x.max(dim=-2)[0]
            # x = text_hidden + audio_hidden
        # x, z_mask = self._get_kuma_mask(audio_hidden, x_mask)
                
        # text_out, _ = self.cross_att(ratio_pooler, audio_x)
        # text_out = F.dropout(text_out)
        
        x_a = self.toprobs(x)
        x_b = self.toprobs_b(x)
        x_a = torch.squeeze(x_a)
        x_b = torch.squeeze(x_b)
        #print('x shape: ',x.shape)
        return x_a, x_b, x, price_hidden
    
    def _get_kuma_mask(self, emb, mask):
        
        z = self.latent_model(emb, mask)
        
        z_mask = (mask.float() * z).unsqueeze(-1)  # [B, T, 1]
        # rnn_mask = z_mask.squeeze(-1) > 0.  # z could be continuous
        # z_mask = (z_mask > 0.5).float()
        emb = emb * z_mask
        
        return emb.mean(1), z_mask
    

class KUMACTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, args, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0, wide=False):

        super().__init__()
        self.args = args
        self.num_tokens, self.max_pool = num_tokens, max_pool

        # self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)
        
        self.ptm = AutoModel.from_pretrained(self.args.ptm)
        
        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)
        
        ####KUMA####
        if self.args.dependent_z:
            self.latent_model = DependentLatentModel(
                embed_size=emb, hidden_size=args.kuma_hidden_size,
                dropout=args.kuma_dropout, layer='lstm')
        else:
            self.latent_model = IndependentLatentModel(
                embed_size=emb, hidden_size=args.kuma_hidden_size,
                dropout=args.kuma_dropout, layer='lstm')

        self.toprobs = nn.Linear(emb, num_classes)

        self.do = nn.Dropout(dropout)

    def forward(self, x, ratio_text_ids, ratio_type_ids, ratio_attention_mask):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        # tokens = self.token_embedding(x)
        tokens = x
        x_mask = (x != 0)
        b, t, e = tokens.size()
        
        ratio_pooler = self.ptm(ratio_text_ids, ratio_type_ids, ratio_attention_mask)['pooler_output']   # 72 *2  *  768
        ratio_pooler = ratio_pooler.view(b, -1, e).mean(1)

        # positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        # x = tokens + positions
        # x = self.do(x)

        # x = self.tblocks(x)

        # x, z_mask = self._get_kuma_mask(x, x_mask)
        # x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        x = ratio_pooler
        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)
    
    def _get_kuma_mask(self, emb, mask):
        
        z = self.latent_model(emb, mask)
        
        z_mask = (mask.float() * z).unsqueeze(-1)  # [B, T, 1]
        # rnn_mask = z_mask.squeeze(-1) > 0.  # z could be continuous
        # z_mask = (z_mask > 0.5).float()
        emb = emb * z_mask
        
        return emb.mean(1), z_mask
    

class RTransformer(nn.Module):
    """
    Transformer for sequences Regression    
    
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0):
        """
        emb: Embedding dimension
        heads: nr. of attention heads
        depth: Number of transformer blocks
        seq_length: Expected maximum sequence length
        num_tokens: Number of tokens (usually words) in the vocabulary
        num_classes: Number of classes.
        max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()

        self.num_tokens, self.max_pool = num_tokens, max_pool

        #self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_classes)
        self.toprobs_b = nn.Linear(emb, num_classes)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        sentences_emb = x
        b, t, e = x.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        #positions = self.pos_embedding(torch.arange(t))[None, :, :].expand(b, t, e)
        #positions = torch.tensor(positions, dtype=torch.float32)
        x = sentences_emb.cuda() + positions
        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension
        
        x_a = self.toprobs(x)
        x_b = self.toprobs_b(x)
        x_a = torch.squeeze(x_a)
        x_b = torch.squeeze(x_b)
        #print('x shape: ',x.shape)
        return x_a, x_b

class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0, wide=False):

        super().__init__()

        self.num_tokens, self.max_pool = num_tokens, max_pool

        # self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_classes)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        # tokens = self.token_embedding(x)
        tokens = x
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)

class PretrainPTM(nn.Module):
    def __init__(self, args):
        super(PretrainPTM, self).__init__()
        self.args = args
        self.ptm = AutoModelForPreTraining.from_pretrained(self.args.ptm, output_hidden_states=True)
        self.pred_is_ratio = nn.Linear(768, 2)
    
    def substi(self, substi_input_ids):
        
        ptm_out = self.ptm(substi_input_ids)
        out = ptm_out['seq_relationship_logits']
        
        return out

    def is_ratio(self, is_ratio_input_ids):
        
        mask = (is_ratio_input_ids != 0).unsqueeze(-1)
        ptm_out = self.ptm(is_ratio_input_ids)
        out = ptm_out['hidden_states'][-1]
        pred = self.pred_is_ratio(out)
        pred = pred * mask
        
        return pred
    
    def mlm(self, mlm_input_ids):
        
        mask = (mlm_input_ids != 0).unsqueeze(-1)
        ptm_out = self.ptm(mlm_input_ids)
        out = ptm_out['prediction_logits']
        pred = out * mask
        
        return pred
    
    def forward(self, substi_input_ids, mlm_input_ids, is_ratio_input_ids, task='substi'):
        
        if task == 'substi':
            pred_substi = self.substi(substi_input_ids)
            return pred_substi
        elif task == 'is_ratio':
            pred_is_ratio = self.is_ratio(is_ratio_input_ids)
            return pred_is_ratio
        elif task == 'mlm':
            pred_mlm = self.mlm(mlm_input_ids)
            return pred_mlm
        
        