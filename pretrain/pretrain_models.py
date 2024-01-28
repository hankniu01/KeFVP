from transformers import AutoModel, AutoModelForPreTraining
import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

class RelationGAT(nn.Module):
    def __init__(self, in_dim, h_dim, head=4):
        super(RelationGAT, self).__init__()
        self.h_dim = h_dim
        self.head = head
        
        self.head_l = nn.Linear(in_dim, h_dim * head)
        self.rel_l = nn.Linear(in_dim, h_dim * head)
        self.tail_l = nn.Linear(in_dim, h_dim * head)
        
        self.relu = nn.ReLU()
        self.rel_gate = nn.Linear(self.h_dim, 1)
    
    def forward(self, head, rel, tail, ent_mask):
        '''
        head: E, T, D
        out: for head B, E, D_out
        '''
        E, T, D = head.shape
        ent_mask = ent_mask.float()
        
        h = self.head_l(head).view(E, T, self.head, self.h_dim).permute(0, 2, 1, 3).contiguous()[:, :, 0, :]  # E, H, d
        r = self.rel_l(rel).view(E, T, self.head, self.h_dim).permute(0, 2, 1, 3).contiguous()
        t = self.tail_l(tail).view(E, T, self.head, self.h_dim).permute(0, 2, 1, 3).contiguous()   # E, H, T, d
        
        ####### relation attention #######
        r = self.relu(r)
        r = self.rel_gate(r)
        r = r.squeeze(-1)   # E, H, T
        r_att = F.softmax(mask_logits(r, ent_mask.unsqueeze(1).repeat(1, self.head, 1)), dim=-1) # E, H, T
        h_out1 = (r_att.unsqueeze(-2) @ t).squeeze(-2)  # E, H, d
        
        ######## normal attention ######
        h_t = (h.unsqueeze(-2) @ t.transpose(-1, -2)).squeeze(-2)  # E, H, T
        att = F.softmax(mask_logits(h_t, ent_mask.unsqueeze(1).repeat(1, self.head, 1)), dim=-1) # E, H, T
        h_out2 = (att.unsqueeze(-2) @ t).squeeze(-2)  # E, H, d
        
        h_out = torch.cat([h, h_out1, h_out2], dim=-1).mean(dim=1)   # E, 3*d
        
        return h_out
        



class PretrainPTMWithKG(nn.Module):
    def __init__(self, args):
        super(PretrainPTMWithKG, self).__init__()
        self.args = args
        self.ptm = AutoModelForPreTraining.from_pretrained(self.args.ptm, output_hidden_states=True)
        self.embedding_size = args.embedding_size
        
        self.lm1 = nn.ModuleList([
            self.ptm.bert.embeddings,
            self.ptm.bert.encoder.layer[0],
            self.ptm.bert.encoder.layer[1],
            self.ptm.bert.encoder.layer[2],
            self.ptm.bert.encoder.layer[3],
            self.ptm.bert.encoder.layer[4],
            self.ptm.bert.encoder.layer[5]
        ])
        
        self.lm2 = nn.ModuleList([
            self.ptm.bert.encoder.layer[6],
            self.ptm.bert.encoder.layer[7],
            self.ptm.bert.encoder.layer[8],
            self.ptm.bert.encoder.layer[9],
            self.ptm.bert.encoder.layer[10],
            self.ptm.bert.encoder.layer[11],
            self.ptm.bert.pooler,
            self.ptm.cls
        ])
        
        self.W_fuse = nn.Linear(self.embedding_size*2, self.embedding_size, bias=False)
        
        self.pred_is_ratio = nn.Linear(self.embedding_size, 2)
        
        self.emb = self.ptm.bert.get_input_embeddings()
        
        self.criterion = nn.MarginRankingLoss(margin=1.0, reduction='none')
    
    def fuse_with_kg(self, ptm_out, head, head_gout, position, ent_mask_for_fuse):
        '''
        position B, L
        ptm_out B, L, D
        head B, E, D
        head_gout B, E, D
        '''
        bidx, pidx = (position == 1).nonzero(as_tuple=True)
        re_head = torch.zeros_like(ptm_out)
        # re_head_gout = torch.zeros_like(ptm_out)   # B, L, D
        re_head[bidx, pidx, :] = head.detach()
        # re_head_gout[bidx, pidx, :] = head_gout.detach()
        
        ptm_out = self.W_fuse(torch.cat([ptm_out, re_head], dim=-1))
        
        return ptm_out
    
    def mlm(self, mlm_input_ids, head, head_gout, position, ent_mask_for_fuse):
        
        b_mask_posi, p_mask_posi = (mlm_input_ids == 5).nonzero(as_tuple=True)   # 5 is mask id
        for idx, layer in enumerate(self.lm1):
            if idx == 0:
                ptm_out = layer(mlm_input_ids)
            else:
                ptm_out = layer(ptm_out)[0]
        ptm_out = self.fuse_with_kg(ptm_out, head, head_gout, position, ent_mask_for_fuse)
        for idx, layer in enumerate(self.lm2):
            if idx != 6 and idx != 7:
                ptm_out = layer(ptm_out)[0]
        pooler_out = self.lm2[6](ptm_out)
        prediction_logits, _ = self.lm2[7](ptm_out, pooler_out)
                
        pred = prediction_logits[b_mask_posi, p_mask_posi, :]
        
        return pred
    
    def forward(self, mlm_input_ids, is_ratio_input_ids, head_ids, rel_ids, tail_ids, postion, posi, task='substi'):
        '''
        position B, E, 2
        '''
        B, E, T, L = head_ids.shape
        ent_mask = (head_ids[:, :, :, 0] != 0) # B, E, T
        ent_mask_for_fuse = (head_ids[:, :, 0, 0] != 0) # B, E
        posi_mask = (posi != -1).float()
        self.bidx, self.pidx1, _ = posi_mask.nonzero(as_tuple=True)
        
        for idx, layer in enumerate(self.lm1):
            if idx == 0:
                head_input = layer(head_ids.view(-1, L).contiguous())
            else:
                head_input = layer(head_input)[0]
        head = head_input.view(B, E, T, L, -1)[self.bidx, self.pidx1, :, 0, :]
        
        pred_mlm = self.mlm(mlm_input_ids, head[:, 0, :], head, postion, ent_mask_for_fuse)
        
        return pred_mlm, 0


