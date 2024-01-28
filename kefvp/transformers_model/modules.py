# from transformers.util import *

import torch
from torch import nn
import torch.nn.functional as F
import copy
from .util import mask_, contains_nan

import random, math
from latent.nn.kuma_gate import KumaGate

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


class CrossFusion(nn.Module):
    
    def __init__(self, h_dim, out_dim=50):
        super().__init__()

        self.lstm = nn.LSTM(h_dim*2, h_dim, bidirectional=True)
    
    def forward(self, t_h, a_h, t_mask, a_mask):
        '''
        text_in: B * L_text * text_indim
        audio_in : B * L_audio * audio_indim
        '''
        # t_h = self.text_linear(text_in)  # B * L_text * h_dim
        # a_h = self.audio_linear(audio_in) # B * L_audio * h_dim
        
        t_a_h = t_h @ a_h.transpose(2, 1)  # B * L_text * L_audio
        t_a_mask = t_mask @ a_mask.transpose(2, 1)
        t_a_h_att = F.softmax(mask_logits(t_a_h, t_a_mask), dim=-1)  # B * L_text * L_audio
        ta_h = t_a_h_att @ a_h   # B * L_text * h_indim
        t_out = t_h + ta_h
        
        a_t_h = a_h @ t_h.transpose(2, 1)
        a_t_mask = a_mask @ t_mask.transpose(2, 1)
        a_t_h_att = F.softmax(mask_logits(a_t_h, a_t_mask), dim=-1)
        at_h = a_t_h_att @ t_h
        a_out = a_h + at_h
        
        # out = torch.cat([t_out, a_out], dim=-1)
        out = t_out + a_out
        # out, _ = self.lstm(out)
        
        return out

class MergeFeature(nn.Module):
    def __init__(self, text_in_dim=200, audio_in_dim=26, h_dim=100, out_dim=50, text_drop=0.5, audio_drop=0.5, ff_hidden_mult=4):
        super(MergeFeature, self).__init__()
        # self.text_lstm = nn.LSTM(text_in_dim, h_dim, bidirectional=True)
        # self.text_drop = nn.Dropout(text_drop)
        self.fc_text = nn.Linear(text_in_dim, h_dim)
        self.text_act = nn.ReLU()

        # self.audio_lstm = nn.LSTM(audio_in_dim, h_dim, bidirectional=True)
        # self.audio_drop = nn.Dropout(audio_drop)
        self.fc_audio = nn.Linear(audio_in_dim, h_dim)
        self.audio_act = nn.ReLU()

        self.crossfuse = CrossFusion(h_dim)
        
        emb = h_dim
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        
        self.do = nn.Dropout(text_drop)
        
        self.fc_out = nn.Linear(emb, out_dim)
        # self.tanh = nn.Tanh()

    def forward(self, x_text, x_audio):
        # mask_text = x_text == 0
        # mask_audio = x_audio == 0
        audio_mask = (x_audio[:, :, 0] != 0).float().unsqueeze(-1)
        text_mask = (x_text[:, :, 0] != 0).float().unsqueeze(-1)
        
        # x_text_ = self.text_drop(self.text_lstm(x_text)[0])
        x_text = self.text_act(self.fc_text(x_text))

        # x_audio_ = self.audio_drop(self.audio_lstm(x_audio)[0])
        x_audio = self.audio_act(self.fc_audio(x_audio))
        
        ##### ablation for multimodal fusion
        out = x_text + x_audio
        
        # fuse_out = self.crossfuse(x_text, x_audio, text_mask, audio_mask)
        # # fuse_out = self.kuma_select(fuse_out)
        # # for i in range(2):
        # #     fuse_out = self.transform[i](fuse_out)
        # # fuse_out = fuse_out.mean(1)

        # # fuse_out = torch.cat([x_text, x_audio], dim=-1)
        
        # x = self.norm1(fuse_out + x_text + x_audio)

        # # x = self.do(x)

        # fedforward = self.ff(x)

        # x = self.norm2(fedforward + x)

        # # out = self.do(x)
        
        # out = self.fc_out(x)
        
        audio_len = audio_mask.nonzero(as_tuple=False).shape[0]
        text_len = text_mask.nonzero(as_tuple=False).shape[0]
        
        if audio_len > text_len:
            mask = audio_mask
        else:
            mask = text_mask

        return out*mask


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

    def forward(self, x, mask_idx):
        
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

        mask_idx = mask_idx.unsqueeze(1).repeat(1, h, 1, 1)
        mask_idx = mask_idx.view(b * h, t, 1).contiguous()
        mask_mask_idx = mask_idx @ mask_idx.transpose(2, 1)
        dot = F.softmax(mask_logits(dot, mask_mask_idx), dim=-1) # dot now has row-wise self-attention probabilities

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
    def __init__(self, emb, heads=8, mask=None, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.heads = heads
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

    def forward(self, x, mask_idx):

        attended = self.attention(x, mask_idx)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x



    
    def __init__(self, num_channel, weights=None):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)  # equal weight
        # if weights != None:
        #     # self.weight.data = nn.Parameter(torch.Tensor(weights).reshape(self.weight.shape))
        #     with torch.no_grad():
        #         w = torch.Tensor(weights).reshape(self.weight.shape)
        #         self.weight.copy_(w)

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        # Row normalization of all graphs generated
        adj_list = F.normalize(adj_list, dim=0, p=1)
        # Hadamard product + summation -> Conv
        w = F.softmax(self.weight, dim=0)
        return torch.sum(adj_list * w, dim=0)
