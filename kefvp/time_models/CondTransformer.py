import torch
import torch.nn as nn
import torch.nn.functional as F
from condlayers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from condlayers.SelfAttention_Family import FullAttention, AttentionLayer
from condlayers.Embed import DataEmbedding
import numpy as np
from condlayers.AutoCorrelation import CondAutoCorrelationLayer, CondAutoCorrelation
from transformers_model.modules import TransformerBlock


class CondTransformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs, output_attention=False):
        super(CondTransformer, self).__init__()
        self.pred_len = configs.duration
        self.output_attention = output_attention
        configs.dec_in = configs.enc_in
        self.configs = configs

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=self.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    self.configs,
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    CondAutoCorrelationLayer(
                        CondAutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    pred_len = self.pred_len
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.cond_embed = nn.Linear(configs.text_indim, configs.d_model)
        nn.init.xavier_normal_(self.cond_embed.weight)
        
        self.cond_att = nn.ModuleList([
            TransformerBlock(configs.d_model),
            TransformerBlock(configs.d_model)
        ])

    def forward(self, x_enc, cond, x_dec=None, x_mark_enc=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        x_dec = x_enc

        enc_out = self.enc_embedding(x_enc, x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        zeros = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]], device=x_enc.device)
        x_dec = torch.cat([x_dec, zeros], dim=1)

        dec_out = self.dec_embedding(x_dec, x_dec)
        
        ###################### for transformer ################
        mask_idx = (cond[:, :, 0] != 0).float().unsqueeze(-1)
        attcond = self.cond_att[0](self.cond_embed(cond), mask_idx)
        
        attcond = self.cond_att[1](attcond, mask_idx)
        
        cond = attcond.mean(1).unsqueeze(1).repeat(1, self.pred_len, 1)
        

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, cond=cond)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :].mean(1)  # [B, L, D]
